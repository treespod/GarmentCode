"""A focused WebGPU XPBD cloth solver for the GarmentCode test path.

The simulation kernels here are WGSL and are dispatched through wgpu-py.  The
implemented constraint set mirrors the cloth path used by GarmentCode's
``Cloth`` object: semi-implicit particle integration, edge springs, and
interior-edge bending constraints.  Mesh broad-phase contacts are handled in a
separate module; this file intentionally contains no Warp dependency.
"""

from __future__ import annotations

import struct
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import wgpu


WORKGROUP_SIZE = 64
_wgpu_instance_configured = False


WGSL = r"""
struct SimParams {
    particle_count: u32,
    spring_count: u32,
    bend_count: u32,
    point_tri_count: u32,
    edge_edge_count: u32,
    _pad_u0: u32,
    _pad_u1: u32,
    _pad_u2: u32,
    dt: f32,
    gravity_x: f32,
    gravity_y: f32,
    gravity_z: f32,
    damping: f32,
    damping_min: f32,
    max_velocity: f32,
    _pad_f0: f32,
    body_enabled: u32,
    body_drag_enabled: u32,
    attachment_frame: u32,
    attachment_frames: u32,
    particle_radius: f32,
    body_mu: f32,
    contact_relaxation: f32,
    _pad_f1: f32,
};

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> old_positions: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> inverse_masses: array<f32>;
@group(0) @binding(5) var<storage, read> spring_indices: array<vec4<u32>>;
@group(0) @binding(6) var<storage, read> spring_properties: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> bend_indices: array<vec4<u32>>;
@group(0) @binding(8) var<storage, read> bend_properties: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read_write> constraint_deltas: array<vec4<f32>>;
@group(0) @binding(10) var<storage, read> incidence_offsets: array<u32>;
@group(0) @binding(11) var<storage, read> incidence_slots: array<u32>;
@group(0) @binding(12) var<storage, read> body_contact_data: array<vec4<f32>>;
@group(0) @binding(13) var<storage, read> point_tri_indices: array<vec4<u32>>;
@group(0) @binding(14) var<storage, read> point_tri_sidedness: array<u32>;
@group(0) @binding(15) var<storage, read> edge_edge_indices: array<vec4<u32>>;
@group(0) @binding(16) var<storage, read> edge_edge_normals: array<vec4<f32>>;
@group(0) @binding(17) var<storage, read> attachment_data: array<vec4<f32>>;
@group(0) @binding(18) var<storage, read> attachment_offsets: array<u32>;

fn closest_triangle_barycentric(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) { return vec3<f32>(1.0, 0.0, 0.0); }

    let bp = p - b;
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) { return vec3<f32>(0.0, 1.0, 0.0); }

    let vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        let v = d1 / (d1 - d3);
        return vec3<f32>(1.0 - v, v, 0.0);
    }

    let cp = p - c;
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) { return vec3<f32>(0.0, 0.0, 1.0); }

    let vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        let w = d2 / (d2 - d6);
        return vec3<f32>(1.0 - w, 0.0, w);
    }

    let va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return vec3<f32>(0.0, 1.0 - w, w);
    }

    let denom = va + vb + vc;
    if (abs(denom) < 1.0e-20) { return vec3<f32>(1.0, 0.0, 0.0); }
    let inv_denom = 1.0 / denom;
    let v = vb * inv_denom;
    let w = vc * inv_denom;
    return vec3<f32>(1.0 - v - w, v, w);
}

fn closest_edge_parameters(a0: vec3<f32>, a1: vec3<f32>, b0: vec3<f32>, b1: vec3<f32>) -> vec2<f32> {
    let d1 = a1 - a0;
    let d2 = b1 - b0;
    let r = a0 - b0;
    let aa = dot(d1, d1);
    let ee = dot(d2, d2);
    let ff = dot(d2, r);
    var s = 0.0;
    var t = 0.0;
    if (aa <= 1.0e-12 && ee <= 1.0e-12) {
        s = 0.0;
        t = 0.0;
    } else if (aa <= 1.0e-12) {
        s = 0.0;
        t = clamp(ff / ee, 0.0, 1.0);
    } else {
        let cc = dot(d1, r);
        if (ee <= 1.0e-12) {
            t = 0.0;
            s = clamp(-cc / aa, 0.0, 1.0);
        } else {
            let bb = dot(d1, d2);
            let denom = aa * ee - bb * bb;
            if (abs(denom) > 1.0e-12) {
                s = clamp((bb * ff - cc * ee) / denom, 0.0, 1.0);
            }
            t = (bb * s + ff) / ee;
            if (t < 0.0) {
                t = 0.0;
                s = clamp(-cc / aa, 0.0, 1.0);
            } else if (t > 1.0) {
                t = 1.0;
                s = clamp((bb - cc) / aa, 0.0, 1.0);
            }
        }
    }
    return vec2<f32>(s, t);
}

@compute @workgroup_size(64)
fn predict_particles(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) { return; }

    let x0 = positions[i].xyz;
    var v1 = velocities[i].xyz;
    old_positions[i] = vec4<f32>(x0, 0.0);

    let inv_mass = inverse_masses[i];
    if (inv_mass > 0.0) {
        v1 = v1 + vec3<f32>(params.gravity_x, params.gravity_y, params.gravity_z) * params.dt;
    }

    // Match Warp's global viscous damping: exp(-damping * |v| * dt), followed
    // by the independent velocity cap.
    let speed_before_damping = length(v1);
    if (params.damping > 0.0 && speed_before_damping > params.damping_min) {
        v1 = v1 * exp(-params.damping * speed_before_damping * params.dt);
    }
    if (params.max_velocity > 0.0 && speed_before_damping > params.max_velocity) {
        let damped_speed = length(v1);
        if (damped_speed > 0.0) {
            v1 = v1 * (params.max_velocity / damped_speed);
        }
    }

    positions[i] = vec4<f32>(x0 + v1 * params.dt, 0.0);
    velocities[i] = vec4<f32>(v1, 0.0);
}

@compute @workgroup_size(64)
fn solve_point_triangle_contacts(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= params.point_tri_count) { return; }
    let ids = point_tri_indices[c];
    let point_id = ids.x;
    let i0 = ids.y;
    let i1 = ids.z;
    let i2 = ids.w;
    let out_slot = 2u * params.spring_count + 4u * params.bend_count + c;
    constraint_deltas[out_slot] = vec4<f32>(0.0);

    let p = positions[point_id].xyz;
    let p0 = positions[i0].xyz;
    let p1 = positions[i1].xyz;
    let p2 = positions[i2].xyz;
    let bary = closest_triangle_barycentric(p, p0, p1, p2);
    let p_closest = bary.x * p0 + bary.y * p1 + bary.z * p2;
    let normal_raw = cross(p1 - p0, p2 - p0);
    let normal_length = length(normal_raw);
    if (normal_length <= 1.0e-12) { return; }
    var n_hat = normal_raw / normal_length;
    if (point_tri_sidedness[c] != 0u) { n_hat = -n_hat; }
    let c_def = dot(n_hat, p - p_closest) - 2.0 * params.particle_radius;
    if (c_def >= 0.0) { return; }

    let a = n_hat.x;
    let b = n_hat.y;
    let z = n_hat.z;
    let a2 = a * a;
    let b2 = b * b;
    let z2 = z * z;
    let nn = vec3<f32>(
        a - a2 * a - a * b2 - a * z2,
        -a2 * b + b - b2 * b - b * z2,
        -a2 * z - b2 * z + z - z2 * z) / normal_length;
    let grad0 = cross(p1 - p2, nn) - n_hat;
    let grad1 = cross(p2 - p0, nn);
    let grad2 = cross(p1 - p0, nn);
    let w0 = inverse_masses[i0];
    let w1 = inverse_masses[i1];
    let w2 = inverse_masses[i2];
    let wp = inverse_masses[point_id];
    let denom = w0 * dot(grad0, grad0) + w1 * dot(grad1, grad1) +
                w2 * dot(grad2, grad2) + wp;
    if (abs(denom) <= 1.0e-5) { return; }
    let dlambda = -c_def / denom;
    constraint_deltas[out_slot] = vec4<f32>(wp * dlambda * n_hat, 0.0);
}

@compute @workgroup_size(64)
fn solve_edge_edge_contacts(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= params.edge_edge_count) { return; }
    let ids = edge_edge_indices[c];
    let ia = ids.x;
    let ib = ids.y;
    let i0 = ids.z;
    let i1 = ids.w;
    let out_base = 2u * params.spring_count + 4u * params.bend_count +
                   params.point_tri_count + 2u * c;
    constraint_deltas[out_base] = vec4<f32>(0.0);
    constraint_deltas[out_base + 1u] = vec4<f32>(0.0);

    let pa = positions[ia].xyz;
    let pb = positions[ib].xyz;
    let e0 = positions[i0].xyz;
    let e1 = positions[i1].xyz;
    let t = closest_edge_parameters(pa, pb, e0, e1);
    let alpha = t.x;
    let beta = t.y;
    let p_alpha = pa * (1.0 - alpha) + pb * alpha;
    let p_beta = e0 * (1.0 - beta) + e1 * beta;
    let separation = p_beta - p_alpha;
    let separation_length = length(separation);
    if (separation_length <= 1.0e-12) { return; }
    var l_dir = separation / separation_length;
    if (dot(l_dir, edge_edge_normals[c].xyz) < 0.0) { l_dir = -l_dir; }
    let constraint = dot(p_beta - p_alpha, l_dir) - 2.0 * params.particle_radius;
    if (constraint >= 0.0) { return; }

    let grad_a = -(1.0 - alpha) * l_dir;
    let grad_b = -alpha * l_dir;
    let grad_0 = -(1.0 - beta) * l_dir;
    let grad_1 = -beta * l_dir;
    let wa = inverse_masses[ia];
    let wb = inverse_masses[ib];
    let w0 = inverse_masses[i0];
    let w1 = inverse_masses[i1];
    let denom = wa * dot(grad_a, grad_a) + wb * dot(grad_b, grad_b) +
                w0 * dot(grad_0, grad_0) + w1 * dot(grad_1, grad_1);
    if (abs(denom) <= 1.0e-5) { return; }
    let dlambda = -constraint / denom;
    constraint_deltas[out_base] = vec4<f32>(wa * dlambda * grad_a, 0.0);
    constraint_deltas[out_base + 1u] = vec4<f32>(wb * dlambda * grad_b, 0.0);
}

@compute @workgroup_size(64)
fn solve_springs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= params.spring_count) { return; }

    let ij = spring_indices[c];
    let i = ij.x;
    let j = ij.y;
    let props = spring_properties[c];
    let rest = props.x;
    let ke = props.y;
    let kd = props.z;

    let out0 = 2u * c;
    let out1 = out0 + 1u;
    constraint_deltas[out0] = vec4<f32>(0.0);
    constraint_deltas[out1] = vec4<f32>(0.0);

    if (ke <= 0.0 || kd < 0.0) { return; }

    let xij = positions[i].xyz - positions[j].xyz;
    let l = length(xij);
    if (l == 0.0) { return; }
    let n = xij / l;
    let wi = inverse_masses[i];
    let wj = inverse_masses[j];
    let denom = wi + wj;
    if (denom <= 0.0) { return; }

    let alpha = 1.0 / (ke * params.dt * params.dt);
    let gamma = kd / (ke * params.dt);
    let c_value = l - rest;
    let grad_c_dot_v = params.dt * dot(n, velocities[i].xyz - velocities[j].xyz);
    let dlambda = -(c_value + gamma * grad_c_dot_v) / ((1.0 + gamma) * denom + alpha);

    constraint_deltas[out0] = vec4<f32>(wi * dlambda * n, 0.0);
    constraint_deltas[out1] = vec4<f32>(-wj * dlambda * n, 0.0);
}

@compute @workgroup_size(64)
fn solve_bending(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= params.bend_count) { return; }

    let base = 2u * params.spring_count + 4u * c;
    constraint_deltas[base + 0u] = vec4<f32>(0.0);
    constraint_deltas[base + 1u] = vec4<f32>(0.0);
    constraint_deltas[base + 2u] = vec4<f32>(0.0);
    constraint_deltas[base + 3u] = vec4<f32>(0.0);

    let ids = bend_indices[c];
    let i = ids.x;
    let j = ids.y;
    let k = ids.z;
    let l = ids.w;
    let props = bend_properties[c];
    let ke = props.x;
    let kd = props.y;
    let rest_angle = props.z;
    if (ke <= 0.0 || kd < 0.0) { return; }

    let x1 = positions[i].xyz;
    let x2 = positions[j].xyz;
    let x3 = positions[k].xyz;
    let x4 = positions[l].xyz;

    var n1 = cross(x3 - x1, x4 - x1);
    var n2 = cross(x4 - x2, x3 - x2);
    let n1_length = length(n1);
    let n2_length = length(n2);
    if (n1_length < 1.0e-6 || n2_length < 1.0e-6) { return; }
    n1 = n1 / n1_length;
    n2 = n2 / n2_length;

    let e = x4 - x3;
    let e_length = length(e);
    if (e_length < 1.0e-8) { return; }
    let e_hat = e / e_length;
    let dot_n = clamp(dot(n1, n2), -1.0, 1.0);
    let sign_value = sign(dot(cross(n1, n2), e)) * -1.0;
    let angle = acos(dot_n);

    let grad_i = n1 * e_length * sign_value;
    let grad_j = n2 * e_length * sign_value;
    let grad_k = (n1 * dot(x1 - x4, e_hat) + n2 * dot(x2 - x4, e_hat)) * sign_value;
    let grad_l = (n1 * dot(x3 - x1, e_hat) + n2 * dot(x3 - x2, e_hat)) * sign_value;

    let wi = inverse_masses[i];
    let wj = inverse_masses[j];
    let wk = inverse_masses[k];
    let wl = inverse_masses[l];
    let denom = wi * dot(grad_i, grad_i) + wj * dot(grad_j, grad_j) +
                wk * dot(grad_k, grad_k) + wl * dot(grad_l, grad_l);
    if (denom <= 0.0) { return; }

    let alpha = 1.0 / (ke * params.dt * params.dt);
    let gamma = kd / (ke * params.dt);
    let grad_dot_v = params.dt * (
        dot(grad_i, velocities[i].xyz) + dot(grad_j, velocities[j].xyz) +
        dot(grad_k, velocities[k].xyz) + dot(grad_l, velocities[l].xyz));
    let dlambda = -((angle - rest_angle) + gamma * grad_dot_v) /
                  ((1.0 + gamma) * denom + alpha);

    constraint_deltas[base + 0u] = vec4<f32>(wi * dlambda * grad_i, 0.0);
    constraint_deltas[base + 1u] = vec4<f32>(wj * dlambda * grad_j, 0.0);
    constraint_deltas[base + 2u] = vec4<f32>(wk * dlambda * grad_k, 0.0);
    constraint_deltas[base + 3u] = vec4<f32>(wl * dlambda * grad_l, 0.0);
}

@compute @workgroup_size(64)
fn gather_and_apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) { return; }

    var delta = vec3<f32>(0.0);

    if (params.attachment_frame < params.attachment_frames) {
        var attachment_id = attachment_offsets[i];
        let attachment_end = attachment_offsets[i + 1u];
        let inv_mass = inverse_masses[i];
        loop {
            if (attachment_id >= attachment_end) { break; }
            let base = 3u * attachment_id;
            let anchor = attachment_data[base].xyz;
            let normal = attachment_data[base + 1u].xyz;
            let stiffness = attachment_data[base + 2u].x;
            let damping = attachment_data[base + 2u].y;
            if (inv_mass > 0.0 && stiffness > 0.0) {
                let constraint = dot(normal, positions[i].xyz + delta - anchor);
                let alpha = 1.0 / (stiffness * params.dt * params.dt);
                let gamma = damping / (stiffness * params.dt);
                let grad_dot_v = params.dt * dot(normal, velocities[i].xyz);
                let dlambda = -(constraint + gamma * grad_dot_v) /
                              ((1.0 + gamma) * inv_mass + alpha);
                delta = delta + inv_mass * dlambda * normal;
            }
            attachment_id = attachment_id + 1u;
        }
    }

    // Static triangle-mesh/body contact anchors are prepared by trimesh on the
    // CPU at the start of the frame. Contact resolution itself stays in WGSL.
    if (params.body_enabled != 0u) {
        let contact = body_contact_data[3u * i + 0u];
        let normal_and_drag = body_contact_data[3u * i + 1u];
        let closest = body_contact_data[3u * i + 2u].xyz;
        let normal = normal_and_drag.xyz;
        if (contact.w > 0.5) {
            let c_body = dot(normal, positions[i].xyz - contact.xyz) - params.particle_radius;
            let inv_mass = inverse_masses[i];
            if (c_body <= 0.0 && inv_mass > 0.0) {
                let v_rel = velocities[i].xyz;
                let vn = dot(normal, v_rel);
                let vt = v_rel - normal * vn;
                let vt_len = length(vt);
                let lambda_f = max(params.body_mu * c_body, -vt_len * params.dt);
                var delta_f = vec3<f32>(0.0);
                if (vt_len > 1.0e-8) {
                    delta_f = (vt / vt_len) * lambda_f;
                }
                let delta_n = normal * c_body;
                delta = delta + (delta_f - delta_n) / inv_mass * params.contact_relaxation;
            }
        }
        if (params.body_drag_enabled != 0u && normal_and_drag.w > 0.5) {
            let err = dot(normal, old_positions[i].xyz - closest);
            if (err < 0.0) {
                delta = delta + (-err) * normal;
            }
        }
    }

    var incidence_id = incidence_offsets[i];
    let incidence_end = incidence_offsets[i + 1u];
    loop {
        if (incidence_id >= incidence_end) { break; }
        delta = delta + constraint_deltas[incidence_slots[incidence_id]].xyz;
        incidence_id = incidence_id + 1u;
    }

    let x_new_unclamped = positions[i].xyz + delta;
    var v_new = (x_new_unclamped - old_positions[i].xyz) / params.dt;
    var x_new = x_new_unclamped;
    let speed = length(v_new);
    if (params.max_velocity > 0.0 && speed > params.max_velocity) {
        v_new = v_new * (params.max_velocity / speed);
        x_new = old_positions[i].xyz + v_new * params.dt;
    }
    positions[i] = vec4<f32>(x_new, 0.0);
    velocities[i] = vec4<f32>(v_new, 0.0);
}
"""


@dataclass
class SolverStats:
    adapter: str
    backend: str
    particles: int
    springs: int
    bends: int
    point_triangle_contacts: int
    edge_edge_contacts: int
    frames: int
    substeps: int
    elapsed_seconds: float


class WgpuXPBDSolver:
    """GPU-resident XPBD cloth solver, with WGSL compute kernels."""

    def __init__(
        self,
        positions: np.ndarray,
        inverse_masses: np.ndarray,
        spring_indices: np.ndarray,
        spring_properties: np.ndarray,
        bend_indices: np.ndarray,
        bend_properties: np.ndarray,
        *,
        velocities: np.ndarray | None = None,
        damping: float = 0.25,
        damping_min_speed: float = 0.0,
        max_velocity: float = 25.0,
        body_contact_data: np.ndarray | None = None,
        attachment_data: np.ndarray | None = None,
        attachment_offsets: np.ndarray | None = None,
        point_tri_indices: np.ndarray | None = None,
        point_tri_sidedness: np.ndarray | None = None,
        edge_edge_indices: np.ndarray | None = None,
        edge_edge_normals: np.ndarray | None = None,
        particle_radius: float = 0.1,
        body_mu: float = 0.5,
        contact_relaxation: float = 0.9,
        body_drag_enabled: bool = False,
        adapter: Any | None = None,
    ) -> None:
        self.initial_positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3).copy()
        self.inverse_masses = np.asarray(inverse_masses, dtype=np.float32).reshape(-1).copy()
        self.spring_indices = np.asarray(spring_indices, dtype=np.uint32).reshape(-1, 4).copy()
        self.spring_properties = np.asarray(spring_properties, dtype=np.float32).reshape(-1, 4).copy()
        self.bend_indices = np.asarray(bend_indices, dtype=np.uint32).reshape(-1, 4).copy()
        self.bend_properties = np.asarray(bend_properties, dtype=np.float32).reshape(-1, 4).copy()

        n = len(self.initial_positions)
        if len(self.inverse_masses) != n:
            raise ValueError("inverse_masses must have one value per vertex")
        if len(self.spring_indices) != len(self.spring_properties):
            raise ValueError("spring index/property counts differ")
        if len(self.bend_indices) != len(self.bend_properties):
            raise ValueError("bend index/property counts differ")
        if velocities is None:
            velocities = np.zeros((n, 3), dtype=np.float32)
        self.initial_velocities = np.asarray(velocities, dtype=np.float32).reshape(-1, 3).copy()
        if len(self.initial_velocities) != n:
            raise ValueError("velocities must have one value per vertex")

        self.damping = float(damping)
        self.damping_min_speed = float(damping_min_speed)
        self.max_velocity = float(max_velocity)
        self.particle_radius = float(particle_radius)
        self.body_mu = float(body_mu)
        self.contact_relaxation = float(contact_relaxation)
        self.body_drag_enabled = bool(body_drag_enabled)
        self.attachment_frames = 0
        self.frame_index = 0
        if body_contact_data is None:
            self.body_contact_data = None
            self.body_enabled = False
        else:
            body_contact_data = np.asarray(body_contact_data, dtype=np.float32).reshape(-1, 3, 4)
            if len(body_contact_data) != n:
                raise ValueError("body_contact_data must have one 3xvec4 record per particle")
            self.body_contact_data = body_contact_data.copy()
            self.body_enabled = True
        if attachment_data is None:
            self.attachment_data = np.empty((0, 3, 4), dtype=np.float32)
        else:
            attachment_data = np.asarray(attachment_data, dtype=np.float32).reshape(-1, 3, 4)
            self.attachment_data = attachment_data.copy()
        if attachment_offsets is None:
            self.attachment_offsets = np.zeros(n + 1, dtype=np.uint32)
        else:
            attachment_offsets = np.asarray(attachment_offsets, dtype=np.uint32).reshape(-1)
            if len(attachment_offsets) != n + 1:
                raise ValueError("attachment_offsets must have one entry per particle plus one")
            if int(attachment_offsets[-1]) != len(self.attachment_data):
                raise ValueError("attachment_offsets do not cover the attachment records")
            self.attachment_offsets = attachment_offsets.copy()
        self.point_tri_indices = np.asarray(
            np.empty((0, 4), dtype=np.uint32) if point_tri_indices is None else point_tri_indices,
            dtype=np.uint32,
        ).reshape(-1, 4).copy()
        self.point_tri_sidedness = np.asarray(
            np.empty(0, dtype=np.uint32) if point_tri_sidedness is None else point_tri_sidedness,
            dtype=np.uint32,
        ).reshape(-1).copy()
        self.edge_edge_indices = np.asarray(
            np.empty((0, 4), dtype=np.uint32) if edge_edge_indices is None else edge_edge_indices,
            dtype=np.uint32,
        ).reshape(-1, 4).copy()
        self.edge_edge_normals = np.asarray(
            np.empty((0, 3), dtype=np.float32) if edge_edge_normals is None else edge_edge_normals,
            dtype=np.float32,
        ).reshape(-1, 3).copy()
        if len(self.point_tri_indices) != len(self.point_tri_sidedness):
            raise ValueError("point-triangle indices/sidedness counts differ")
        if len(self.edge_edge_indices) != len(self.edge_edge_normals):
            raise ValueError("edge-edge indices/normals counts differ")
        self.n = n
        self.ns = len(self.spring_indices)
        self.nb = len(self.bend_indices)
        self.npt = len(self.point_tri_indices)
        self.nee = len(self.edge_edge_indices)

        self._build_incidence()
        self._buffer_capacities: dict[str, int] = {}
        self.adapter = adapter or self._choose_adapter()
        self.device = self.adapter.request_device_sync(
            label="GarmentCode WGPU XPBD",
            required_features=(),
        )
        self.queue = self.device.queue
        self.adapter_info = self.adapter.info
        self._create_resources()
        self.positions = self.initial_positions.copy()
        self.velocities = self.initial_velocities.copy()
        self.last_stats: SolverStats | None = None

    @staticmethod
    def _choose_adapter():
        """Prefer a hardware adapter, then fall back to a WebGPU software adapter."""
        global _wgpu_instance_configured
        if not _wgpu_instance_configured:
            if sys.platform.startswith("linux"):
                from wgpu.backends.wgpu_native.extras import set_instance_extras

                set_instance_extras(backends=["Vulkan"])
            _wgpu_instance_configured = True

        try:
            adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        except Exception:
            adapter = None
        if adapter is None:
            adapter = wgpu.gpu.request_adapter_sync(
                power_preference="low-power", force_fallback_adapter=True
            )
        if adapter is None:
            raise RuntimeError("No WebGPU adapter was found (Vulkan/Metal/D3D12 required).")
        return adapter

    def _build_incidence(self) -> None:
        incident: list[list[int]] = [[] for _ in range(self.n)]
        spring_base = 0
        bend_base = 2 * self.ns
        point_tri_base = bend_base + 4 * self.nb
        edge_edge_base = point_tri_base + self.npt

        # Keep contact terms first in each particle's gather order, followed by
        # springs and bending, matching Warp's per-particle correction order.
        for cid, ids in enumerate(self.point_tri_indices):
            incident[int(ids[0])].append(point_tri_base + cid)
        for cid, ids in enumerate(self.edge_edge_indices):
            incident[int(ids[0])].append(edge_edge_base + 2 * cid)
            incident[int(ids[1])].append(edge_edge_base + 2 * cid + 1)
        for cid, pair in enumerate(self.spring_indices):
            incident[int(pair[0])].append(spring_base + 2 * cid)
            incident[int(pair[1])].append(spring_base + 2 * cid + 1)
        for cid, quad in enumerate(self.bend_indices):
            for corner, vertex in enumerate(quad):
                incident[int(vertex)].append(bend_base + 4 * cid + corner)

        offsets = np.zeros(self.n + 1, dtype=np.uint32)
        for i, row in enumerate(incident):
            offsets[i + 1] = offsets[i] + len(row)
        slots = np.fromiter((slot for row in incident for slot in row), dtype=np.uint32)
        self.incidence_offsets = offsets
        self.incidence_slots = slots
        self.constraint_delta_count = edge_edge_base + 2 * self.nee

    def _storage_buffer(self, data: np.ndarray, label: str):
        data = np.ascontiguousarray(data)
        raw = data.tobytes()
        size = max(len(raw), 4)
        buffer = self.device.create_buffer(
            label=label,
            size=size,
            usage=wgpu.BufferUsage.STORAGE
            | wgpu.BufferUsage.COPY_DST
            | wgpu.BufferUsage.COPY_SRC,
        )
        if raw:
            self.queue.write_buffer(buffer, 0, raw)
        self._buffer_capacities[label] = size
        return buffer

    def _upload_storage_buffer(
        self, attribute: str, data: np.ndarray, label: str
    ) -> bool:
        data = np.ascontiguousarray(data)
        raw = data.tobytes()
        buffer = getattr(self, attribute)
        required_size = max(len(raw), 4)
        if required_size > self._buffer_capacities[label]:
            buffer = self._storage_buffer(data, label)
            setattr(self, attribute, buffer)
            return True
        if raw:
            self.queue.write_buffer(buffer, 0, raw)
        return False

    def update_self_contact_data(
        self,
        point_tri_indices: np.ndarray,
        point_tri_sidedness: np.ndarray,
        edge_edge_indices: np.ndarray,
        edge_edge_normals: np.ndarray,
    ) -> None:
        point_tri_indices = np.asarray(point_tri_indices, dtype=np.uint32).reshape(-1, 4)
        point_tri_sidedness = np.asarray(point_tri_sidedness, dtype=np.uint32).reshape(-1)
        edge_edge_indices = np.asarray(edge_edge_indices, dtype=np.uint32).reshape(-1, 4)
        edge_edge_normals = np.asarray(edge_edge_normals, dtype=np.float32).reshape(-1, 3)
        if len(point_tri_indices) != len(point_tri_sidedness):
            raise ValueError("point-triangle indices/sidedness counts differ")
        if len(edge_edge_indices) != len(edge_edge_normals):
            raise ValueError("edge-edge indices/normals counts differ")
        if point_tri_indices.size and int(point_tri_indices.max()) >= self.n:
            raise ValueError("point-triangle contact index is out of range")
        if edge_edge_indices.size and int(edge_edge_indices.max()) >= self.n:
            raise ValueError("edge-edge contact index is out of range")

        self.point_tri_indices = point_tri_indices.copy()
        self.point_tri_sidedness = point_tri_sidedness.copy()
        self.edge_edge_indices = edge_edge_indices.copy()
        self.edge_edge_normals = edge_edge_normals.copy()
        self.npt = len(self.point_tri_indices)
        self.nee = len(self.edge_edge_indices)
        self._build_incidence()

        point_tri_buffer_changed = self._upload_storage_buffer(
            "point_tri_index_buffer",
            _as_vec4_indices(self.point_tri_indices, 4),
            "point-triangle contacts",
        )
        sidedness_buffer_changed = self._upload_storage_buffer(
            "point_tri_sidedness_buffer",
            self.point_tri_sidedness,
            "point-triangle sidedness",
        )
        edge_index_buffer_changed = self._upload_storage_buffer(
            "edge_edge_index_buffer",
            _as_vec4_indices(self.edge_edge_indices, 4),
            "edge-edge contacts",
        )
        edge_normal_buffer_changed = self._upload_storage_buffer(
            "edge_edge_normal_buffer",
            self._vec4(self.edge_edge_normals),
            "edge-edge contact normals",
        )
        delta = np.zeros((max(self.constraint_delta_count, 1), 4), dtype=np.float32)
        delta_buffer_changed = self._upload_storage_buffer(
            "constraint_delta_buffer", delta, "constraint deltas"
        )
        slot_buffer_changed = self._upload_storage_buffer(
            "incidence_slot_buffer",
            self.incidence_slots,
            "incidence slots",
        )
        self._upload_storage_buffer(
            "incidence_offset_buffer",
            self.incidence_offsets,
            "incidence offsets",
        )

        if point_tri_buffer_changed or sidedness_buffer_changed or delta_buffer_changed:
            self.point_tri_bind_group = self._bind_group(
                self.point_tri_pipeline,
                {0: self.params_buffer, 1: self.q_buffer, 4: self.inv_mass_buffer,
                 9: self.constraint_delta_buffer, 13: self.point_tri_index_buffer,
                 14: self.point_tri_sidedness_buffer},
            )
        if edge_index_buffer_changed or edge_normal_buffer_changed or delta_buffer_changed:
            self.edge_edge_bind_group = self._bind_group(
                self.edge_edge_pipeline,
                {0: self.params_buffer, 1: self.q_buffer, 4: self.inv_mass_buffer,
                 9: self.constraint_delta_buffer, 15: self.edge_edge_index_buffer,
                 16: self.edge_edge_normal_buffer},
            )
        if delta_buffer_changed:
            self.spring_bind_group = self._bind_group(
                self.spring_pipeline,
                {0: self.params_buffer, 1: self.q_buffer, 2: self.v_buffer,
                 4: self.inv_mass_buffer, 5: self.spring_index_buffer,
                 6: self.spring_properties_buffer, 9: self.constraint_delta_buffer},
            )
            self.bend_bind_group = self._bind_group(
                self.bend_pipeline,
                {0: self.params_buffer, 1: self.q_buffer, 2: self.v_buffer,
                 4: self.inv_mass_buffer, 7: self.bend_index_buffer,
                 8: self.bend_properties_buffer, 9: self.constraint_delta_buffer},
            )
        if delta_buffer_changed or slot_buffer_changed:
            self.apply_bind_group = self._bind_group(
                self.apply_pipeline,
                {0: self.params_buffer, 1: self.q_buffer, 2: self.v_buffer,
                 3: self.old_q_buffer, 4: self.inv_mass_buffer,
                 9: self.constraint_delta_buffer, 10: self.incidence_offset_buffer,
                 11: self.incidence_slot_buffer, 12: self.body_contact_buffer,
                 17: self.attachment_buffer, 18: self.attachment_offset_buffer},
            )

    @staticmethod
    def _vec4(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32).reshape(-1, 3)
        out = np.zeros((len(values), 4), dtype=np.float32)
        out[:, :3] = values
        return out

    def _create_resources(self) -> None:
        self.q_buffer = self._storage_buffer(self._vec4(self.initial_positions), "positions")
        self.v_buffer = self._storage_buffer(self._vec4(self.initial_velocities), "velocities")
        self.old_q_buffer = self._storage_buffer(np.zeros((self.n, 4), dtype=np.float32), "old positions")
        self.inv_mass_buffer = self._storage_buffer(self.inverse_masses, "inverse masses")

        spring_indices = np.zeros((max(self.ns, 1), 4), dtype=np.uint32)
        spring_props = np.zeros((max(self.ns, 1), 4), dtype=np.float32)
        if self.ns:
            spring_indices[: self.ns, :2] = self.spring_indices[:, :2]
            spring_props[: self.ns] = self.spring_properties
        self.spring_index_buffer = self._storage_buffer(spring_indices, "spring indices")
        self.spring_properties_buffer = self._storage_buffer(spring_props, "spring properties")

        bend_indices = np.zeros((max(self.nb, 1), 4), dtype=np.uint32)
        bend_props = np.zeros((max(self.nb, 1), 4), dtype=np.float32)
        if self.nb:
            bend_indices[: self.nb] = self.bend_indices
            bend_props[: self.nb] = self.bend_properties
        self.bend_index_buffer = self._storage_buffer(bend_indices, "bend indices")
        self.bend_properties_buffer = self._storage_buffer(bend_props, "bend properties")

        point_tri_indices = np.zeros((max(self.npt, 1), 4), dtype=np.uint32)
        point_tri_sidedness = np.zeros(max(self.npt, 1), dtype=np.uint32)
        edge_edge_indices = np.zeros((max(self.nee, 1), 4), dtype=np.uint32)
        edge_edge_normals = np.zeros((max(self.nee, 1), 4), dtype=np.float32)
        if self.npt:
            point_tri_indices[: self.npt] = self.point_tri_indices
            point_tri_sidedness[: self.npt] = self.point_tri_sidedness
        if self.nee:
            edge_edge_indices[: self.nee] = self.edge_edge_indices
            edge_edge_normals[: self.nee, :3] = self.edge_edge_normals
        self.point_tri_index_buffer = self._storage_buffer(point_tri_indices, "point-triangle contacts")
        self.point_tri_sidedness_buffer = self._storage_buffer(point_tri_sidedness, "point-triangle sidedness")
        self.edge_edge_index_buffer = self._storage_buffer(edge_edge_indices, "edge-edge contacts")
        self.edge_edge_normal_buffer = self._storage_buffer(edge_edge_normals, "edge-edge contact normals")

        delta = np.zeros((max(self.constraint_delta_count, 1), 4), dtype=np.float32)
        self.constraint_delta_buffer = self._storage_buffer(delta, "constraint deltas")
        self.incidence_offset_buffer = self._storage_buffer(self.incidence_offsets, "incidence offsets")
        self.incidence_slot_buffer = self._storage_buffer(self.incidence_slots, "incidence slots")
        body_data = self.body_contact_data
        if body_data is None:
            body_data = np.zeros((self.n, 3, 4), dtype=np.float32)
        self.body_contact_buffer = self._storage_buffer(
            body_data.reshape(-1, 4), "body contact anchors"
        )
        attachment_buffer_data = (
            self.attachment_data.reshape(-1, 4)
            if len(self.attachment_data)
            else np.zeros((3, 4), dtype=np.float32)
        )
        self.attachment_buffer = self._storage_buffer(
            attachment_buffer_data, "attachment constraints"
        )
        self.attachment_offset_buffer = self._storage_buffer(
            self.attachment_offsets, "attachment offsets"
        )

        # Uniform block: six 16-byte lanes (96 bytes total).
        self.params_buffer = self.device.create_buffer(
            label="simulation parameters",
            size=96,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        shader = self.device.create_shader_module(label="garment XPBD WGSL", code=WGSL)
        self.predict_pipeline = self.device.create_compute_pipeline(
            label="predict particles", layout="auto",
            compute={"module": shader, "entry_point": "predict_particles"},
        )
        self.spring_pipeline = self.device.create_compute_pipeline(
            label="solve springs", layout="auto",
            compute={"module": shader, "entry_point": "solve_springs"},
        )
        self.bend_pipeline = self.device.create_compute_pipeline(
            label="solve bending", layout="auto",
            compute={"module": shader, "entry_point": "solve_bending"},
        )
        self.point_tri_pipeline = self.device.create_compute_pipeline(
            label="solve point-triangle self contacts", layout="auto",
            compute={"module": shader, "entry_point": "solve_point_triangle_contacts"},
        )
        self.edge_edge_pipeline = self.device.create_compute_pipeline(
            label="solve edge-edge self contacts", layout="auto",
            compute={"module": shader, "entry_point": "solve_edge_edge_contacts"},
        )
        self.apply_pipeline = self.device.create_compute_pipeline(
            label="gather and apply", layout="auto",
            compute={"module": shader, "entry_point": "gather_and_apply"},
        )

        self.predict_bind_group = self._bind_group(
            self.predict_pipeline,
            {0: self.params_buffer, 1: self.q_buffer, 2: self.v_buffer,
             3: self.old_q_buffer, 4: self.inv_mass_buffer},
        )
        self.spring_bind_group = self._bind_group(
            self.spring_pipeline,
            {0: self.params_buffer, 1: self.q_buffer, 2: self.v_buffer,
             4: self.inv_mass_buffer, 5: self.spring_index_buffer,
             6: self.spring_properties_buffer, 9: self.constraint_delta_buffer},
        )
        self.bend_bind_group = self._bind_group(
            self.bend_pipeline,
            {0: self.params_buffer, 1: self.q_buffer, 2: self.v_buffer,
             4: self.inv_mass_buffer, 7: self.bend_index_buffer,
             8: self.bend_properties_buffer, 9: self.constraint_delta_buffer},
        )
        self.point_tri_bind_group = self._bind_group(
            self.point_tri_pipeline,
            {0: self.params_buffer, 1: self.q_buffer, 4: self.inv_mass_buffer,
             9: self.constraint_delta_buffer, 13: self.point_tri_index_buffer,
             14: self.point_tri_sidedness_buffer},
        )
        self.edge_edge_bind_group = self._bind_group(
            self.edge_edge_pipeline,
            {0: self.params_buffer, 1: self.q_buffer, 4: self.inv_mass_buffer,
             9: self.constraint_delta_buffer, 15: self.edge_edge_index_buffer,
             16: self.edge_edge_normal_buffer},
        )
        self.apply_bind_group = self._bind_group(
            self.apply_pipeline,
            {0: self.params_buffer, 1: self.q_buffer, 2: self.v_buffer,
             3: self.old_q_buffer, 4: self.inv_mass_buffer,
             9: self.constraint_delta_buffer, 10: self.incidence_offset_buffer,
             11: self.incidence_slot_buffer, 12: self.body_contact_buffer,
             17: self.attachment_buffer, 18: self.attachment_offset_buffer},
        )

    def _bind_group(self, pipeline, resources: dict[int, Any]):
        return self.device.create_bind_group(
            layout=pipeline.get_bind_group_layout(0),
            entries=[{"binding": binding, "resource": buffer}
                     for binding, buffer in sorted(resources.items())],
        )

    def _update_params(self, dt: float, gravity: tuple[float, float, float]) -> None:
        payload = struct.pack(
            "<8I8f4I4f",
            self.n, self.ns, self.nb, self.npt, self.nee, 0, 0, 0,
            float(dt), float(gravity[0]), float(gravity[1]), float(gravity[2]),
            self.damping, self.damping_min_speed, self.max_velocity, 0.0,
            int(self.body_enabled), int(self.body_drag_enabled),
            int(self.frame_index), int(self.attachment_frames),
            self.particle_radius, self.body_mu, self.contact_relaxation,
            0.0,
        )
        self.queue.write_buffer(self.params_buffer, 0, payload)

    def _dispatch(self, encoder, pipeline, bind_group, count: int, label: str) -> None:
        if count <= 0:
            return
        compute_pass = encoder.begin_compute_pass(label=label)
        compute_pass.set_pipeline(pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups((count + WORKGROUP_SIZE - 1) // WORKGROUP_SIZE)
        compute_pass.end()

    def step_frame(
        self,
        *,
        dt: float = 1.0 / 60.0,
        substeps: int = 10,
        gravity: tuple[float, float, float] = (0.0, -9.81, 0.0),
    ) -> None:
        if substeps < 1:
            raise ValueError("substeps must be >= 1")
        sub_dt = float(dt) / int(substeps)
        self._update_params(sub_dt, gravity)
        encoder = self.device.create_command_encoder(label="garment frame")
        for _ in range(substeps):
            self._dispatch(encoder, self.predict_pipeline, self.predict_bind_group,
                           self.n, "predict particles")
            self._dispatch(encoder, self.point_tri_pipeline, self.point_tri_bind_group,
                           self.npt, "solve point-triangle self contacts")
            self._dispatch(encoder, self.edge_edge_pipeline, self.edge_edge_bind_group,
                           self.nee, "solve edge-edge self contacts")
            self._dispatch(encoder, self.spring_pipeline, self.spring_bind_group,
                           self.ns, "solve springs")
            self._dispatch(encoder, self.bend_pipeline, self.bend_bind_group,
                           self.nb, "solve bending")
            self._dispatch(encoder, self.apply_pipeline, self.apply_bind_group,
                           self.n, "gather and apply")
        self.queue.submit([encoder.finish()])

    def simulate(
        self,
        *,
        frames: int,
        fps: float = 60.0,
        substeps: int = 10,
        gravity: tuple[float, float, float] = (0.0, -9.81, 0.0),
        zero_gravity_steps: int = 0,
        attachment_frames: int = 0,
        body_contact_updater: Callable[[np.ndarray], np.ndarray] | None = None,
        self_contact_updater: Callable[
            [np.ndarray, np.ndarray, int],
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        ] | None = None,
    ) -> SolverStats:
        t0 = time.perf_counter()
        self.attachment_frames = max(int(attachment_frames), 0)
        for frame in range(int(frames)):
            self.frame_index = frame
            if frame > 0 and (
                body_contact_updater is not None or self_contact_updater is not None
            ):
                current = self._read_positions_from_device()
                self.positions = current
                if body_contact_updater is not None:
                    self.update_body_contact_data(body_contact_updater(current))
                if self_contact_updater is not None:
                    velocities = self._read_velocities_from_device()
                    self.velocities = velocities
                    self.update_self_contact_data(
                        *self_contact_updater(current, velocities, frame)
                    )
            frame_gravity = (0.0, 0.0, 0.0) if frame < zero_gravity_steps else gravity
            self.step_frame(dt=1.0 / float(fps), substeps=substeps, gravity=frame_gravity)
        # queue.read_buffer waits for all submitted work, and gives us a stable
        # host copy for OBJ output/comparison.
        self.positions = np.frombuffer(
            self.queue.read_buffer(self.q_buffer), dtype=np.float32,
            count=self.n * 4,
        ).reshape(self.n, 4)[:, :3].copy()
        self.velocities = np.frombuffer(
            self.queue.read_buffer(self.v_buffer), dtype=np.float32,
            count=self.n * 4,
        ).reshape(self.n, 4)[:, :3].copy()
        elapsed = time.perf_counter() - t0
        info = self.adapter_info
        self.last_stats = SolverStats(
            adapter=str(info.get("device", "unknown")),
            backend=str(info.get("backend_type", "unknown")),
            particles=self.n,
            springs=self.ns,
            bends=self.nb,
            point_triangle_contacts=self.npt,
            edge_edge_contacts=self.nee,
            frames=int(frames),
            substeps=int(substeps),
            elapsed_seconds=elapsed,
        )
        return self.last_stats

    def _read_positions_from_device(self) -> np.ndarray:
        return np.frombuffer(
            self.queue.read_buffer(self.q_buffer), dtype=np.float32,
            count=self.n * 4,
        ).reshape(self.n, 4)[:, :3].copy()

    def _read_velocities_from_device(self) -> np.ndarray:
        return np.frombuffer(
            self.queue.read_buffer(self.v_buffer), dtype=np.float32,
            count=self.n * 4,
        ).reshape(self.n, 4)[:, :3].copy()

    def update_body_contact_data(self, data: np.ndarray) -> None:
        data = np.asarray(data, dtype=np.float32).reshape(-1, 3, 4)
        if len(data) != self.n:
            raise ValueError("body contact updater must return one 3xvec4 record per particle")
        self.body_contact_data = data.copy()
        self.body_enabled = True
        self.queue.write_buffer(self.body_contact_buffer, 0, data.reshape(-1, 4))

    def current_positions(self) -> np.ndarray:
        if self.positions is None:
            self.positions = self._read_positions_from_device()
        return np.asarray(self.positions, dtype=np.float32)


def _as_vec4_indices(values: np.ndarray, columns: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.uint32).reshape(-1, columns)
    out = np.zeros((len(values), 4), dtype=np.uint32)
    out[:, :columns] = values
    return out


def make_spring_arrays(
    spring_indices: np.ndarray,
    rest_lengths: np.ndarray,
    stiffness: float,
    damping: float,
) -> tuple[np.ndarray, np.ndarray]:
    idx = _as_vec4_indices(spring_indices, 2)
    props = np.zeros((len(idx), 4), dtype=np.float32)
    if len(idx):
        props[:, 0] = np.asarray(rest_lengths, dtype=np.float32)
        props[:, 1] = float(stiffness)
        props[:, 2] = float(damping)
    return idx, props


def make_bend_arrays(
    bend_indices: np.ndarray,
    stiffness: float,
    damping: float,
    rest_angle: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    idx = _as_vec4_indices(bend_indices, 4)
    props = np.zeros((len(idx), 4), dtype=np.float32)
    if len(idx):
        props[:, 0] = float(stiffness)
        props[:, 1] = float(damping)
        props[:, 2] = float(rest_angle)
    return idx, props
