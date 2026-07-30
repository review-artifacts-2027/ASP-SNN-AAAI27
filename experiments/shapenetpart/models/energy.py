"""
models/energy.py — Energy accounting for the ASP-SNN pipeline.

Implements the standard SNN energy model:
    Analog/MAC layers:   E = FLOPs * E_MAC
    Spiking/AC layers:   E = SOPs  * E_AC,   SOPs = firing_rate * T * FLOPs

Per-operation energy at 45nm (Horowitz, ISSCC 2014):
    E_MAC = 4.6 pJ   (32-bit float multiply-accumulate)
    E_AC  = 0.9 pJ   (32-bit float accumulate)

Batch 4 update — added seg head accounting:
    The per-point seg head runs on N points per sample and its cost is
    substantial (~15–25% of total FLOPs at ShapeNetPart's config). It's
    now counted as either analog or spiking depending on `seg_head_type`.
    This is the change that makes the system-level α number honest at the
    seg-head boundary — previously seg head was silently analog even when
    the encoder was spiked.

Batch 3 update — added fine encoder accounting:
    When use_multiscale=True, a second, smaller encoder is added.
    Included at analog cost (fine encoder stays analog in Batch 4; will
    be spiked in a later batch if we choose to).
"""

E_MAC = 4.6e-12  # joules (4.6 pJ)
E_AC  = 0.9e-12  # joules (0.9 pJ)


def estimate_flops(cfg) -> dict:
    """
    FLOP (MAC-count) estimate per pipeline component for one sample.
    Order-of-magnitude values, sufficient for reported energy ratios.
    """
    M      = getattr(cfg, 'num_slices', 16)
    K      = getattr(cfg, 'points_per_slice', 128)
    k_edge = getattr(cfg, 'k_edge', 20)
    feat   = getattr(cfg, 'feat_dim', 512)
    hidden = getattr(cfg, 'hidden_dim', 512)
    ffn    = getattr(cfg, 'transformer_ffn_dim', 1024)
    n_lif  = getattr(cfg, 'num_lif_layers', 3)
    T      = getattr(cfg, 'T', 6)
    in_ch  = getattr(cfg, 'in_channels', 6)

    # ── Coarse encoder ────────────────────────────────────────────────
    # EdgeConv: per slice, per point, per edge: Conv2d(2C -> 128 -> 128)
    # then Conv1d(128 -> 256 -> feat).
    edgeconv       = M * K * k_edge * (2 * in_ch * 128 + 128 * 128)
    conv1d         = M * K * (128 * 256 + 256 * feat)
    encoder_macs   = edgeconv + conv1d

    # ── Transformer + pos (analog, always) ────────────────────────────
    transformer_macs = M * M * feat + M * feat * ffn * 2
    pos_macs         = M * 3 * feat

    # ── LIF head (ASP loop) ───────────────────────────────────────────
    lif_flops = T * n_lif * hidden * hidden

    # ── B4: Fine encoder (analog for now) ─────────────────────────────
    use_multiscale = getattr(cfg, 'use_multiscale', True)
    fine_encoder_macs = 0.0
    if use_multiscale:
        M_f      = getattr(cfg, 'num_slices_fine', 64)
        K_f      = getattr(cfg, 'points_per_slice_fine', 32)
        k_f      = getattr(cfg, 'fine_k_edge', 8)
        feat_f   = getattr(cfg, 'fine_feat_dim', 128)
        fine_edgeconv = M_f * K_f * k_f * (2 * in_ch * 128 + 128 * 128)
        fine_conv1d   = M_f * K_f * (128 * 256 + 256 * feat_f)
        fine_encoder_macs = fine_edgeconv + fine_conv1d

    # ── B1: Boundary head (tiny, analog) ──────────────────────────────
    use_bnd = getattr(cfg, 'use_boundary_aware', True)
    boundary_macs = 0.0
    if use_bnd:
        N = getattr(cfg, 'num_points', 2048)
        bnd_hidden = getattr(cfg, 'boundary_hidden', 128)
        boundary_macs = N * (feat * bnd_hidden + bnd_hidden * 1)

    # ── B2: Seg head — analog or spiking ──────────────────────────────
    # Seg head input dim: local(feat)+global(feat)+point(64)+cat+xyz(3)+fine
    N            = getattr(cfg, 'num_points', 2048)
    point_dim    = getattr(cfg, 'point_feat_dim', 64)
    xyz_dim      = 3
    fine_dim     = getattr(cfg, 'fine_feat_dim', 128) if use_multiscale else 0
    num_cats     = getattr(cfg, 'num_categories', 0) if getattr(cfg, 'use_category', False) else 0
    seg_in_dim   = feat * 2 + point_dim + num_cats + xyz_dim + fine_dim
    # 3 hidden layers: in→256, 256→256, 256→128, then classifier 128→C
    num_classes  = getattr(cfg, 'num_classes', 50)
    seg_head_flops = N * (
        seg_in_dim * 256 + 256 * 256 + 256 * 128 + 128 * num_classes
    )

    return {
        'encoder_macs':      float(encoder_macs),
        'fine_encoder_macs': float(fine_encoder_macs),
        'transformer_analog': float(transformer_macs + pos_macs),
        'boundary_macs':     float(boundary_macs),
        'lif_head_flops':    float(lif_flops),
        'seg_head_flops':    float(seg_head_flops),
    }


def compute_energy(cfg, mean_firing_rate_head: float,
                   mean_firing_rate_encoder: float = None,
                   mean_firing_rate_seg_head: float = None) -> dict:
    """
    Compute system-level energy estimate (joules per sample).

    Args:
        cfg: config
        mean_firing_rate_head: LIF head firing rate [0,1]
        mean_firing_rate_encoder: spiking encoder firing rate [0,1]. Ignored
            when encoder_type=='analog'.
        mean_firing_rate_seg_head: spiking seg head firing rate [0,1] (Batch 4).
            Ignored when seg_head_type=='analog'. When None with a spiking
            head, defaults to 0.2 (typical measurement in Yao et al. Spike-
            driven Transformer V2) so the report can still be generated.
    """
    flops         = estimate_flops(cfg)
    T_enc         = getattr(cfg, 'encoder_T', 4)
    T_seg         = getattr(cfg, 'seg_head_T', 2)
    encoder_type  = getattr(cfg, 'encoder_type', 'analog')
    seg_head_type = getattr(cfg, 'seg_head_type', 'spiking')

    # ── Coarse encoder ────────────────────────────────────────────────
    if encoder_type == 'spiking' and mean_firing_rate_encoder is not None:
        e_encoder     = flops['encoder_macs'] * mean_firing_rate_encoder * T_enc * E_AC
        e_encoder_ann = flops['encoder_macs'] * E_MAC
    else:
        e_encoder     = flops['encoder_macs'] * E_MAC
        e_encoder_ann = flops['encoder_macs'] * E_MAC

    # ── Fine encoder (always analog in Batch 4) ───────────────────────
    e_fine_encoder     = flops['fine_encoder_macs'] * E_MAC
    e_fine_encoder_ann = flops['fine_encoder_macs'] * E_MAC

    # ── Transformer + pos ─────────────────────────────────────────────
    e_transformer = flops['transformer_analog'] * E_MAC

    # ── Boundary head (small, analog) ─────────────────────────────────
    e_boundary = flops['boundary_macs'] * E_MAC

    # ── LIF head ──────────────────────────────────────────────────────
    e_lif_snn = flops['lif_head_flops'] * mean_firing_rate_head * E_AC
    e_lif_ann = flops['lif_head_flops'] * E_MAC

    # ── B2: Seg head — analog OR spiking (Batch 4) ────────────────────
    if seg_head_type == 'spiking':
        fr_seg = mean_firing_rate_seg_head if mean_firing_rate_seg_head is not None else 0.20
        e_seg_head     = flops['seg_head_flops'] * fr_seg * T_seg * E_AC
        e_seg_head_ann = flops['seg_head_flops'] * E_MAC
    else:
        e_seg_head     = flops['seg_head_flops'] * E_MAC
        e_seg_head_ann = flops['seg_head_flops'] * E_MAC
        fr_seg = None

    # ── Totals ────────────────────────────────────────────────────────
    e_asp_total = (e_encoder + e_fine_encoder + e_transformer
                   + e_boundary + e_lif_snn + e_seg_head)
    e_ann_equiv = (e_encoder_ann + e_fine_encoder_ann + e_transformer
                   + e_boundary + e_lif_ann + e_seg_head_ann)

    alpha_system = e_ann_equiv / max(e_asp_total, 1e-30)
    alpha_head   = e_lif_ann / max(e_lif_snn, 1e-30)
    alpha_encoder = (e_encoder_ann / max(e_encoder, 1e-30)
                     if encoder_type == 'spiking' else 1.0)
    alpha_seg_head = (e_seg_head_ann / max(e_seg_head, 1e-30)
                      if seg_head_type == 'spiking' else 1.0)

    # Fraction of compute that is spiking
    spiking_ops = flops['lif_head_flops']
    total_ops   = (flops['encoder_macs']
                   + flops['fine_encoder_macs']
                   + flops['transformer_analog']
                   + flops['boundary_macs']
                   + flops['lif_head_flops']
                   + flops['seg_head_flops'])
    if encoder_type == 'spiking':
        spiking_ops += flops['encoder_macs'] * T_enc
        total_ops   += flops['encoder_macs'] * (T_enc - 1)
    if seg_head_type == 'spiking':
        spiking_ops += flops['seg_head_flops'] * T_seg
        total_ops   += flops['seg_head_flops'] * (T_seg - 1)
    spiking_frac = spiking_ops / max(total_ops, 1)

    return {
        'encoder_type':               encoder_type,
        'encoder_T':                  T_enc if encoder_type == 'spiking' else 1,
        'seg_head_type':              seg_head_type,
        'seg_head_T':                 T_seg if seg_head_type == 'spiking' else 1,
        'mean_firing_rate_head':      mean_firing_rate_head,
        'mean_firing_rate_encoder':   mean_firing_rate_encoder,
        'mean_firing_rate_seg_head':  fr_seg,
        'e_encoder_pJ':               e_encoder * 1e12,
        'e_fine_encoder_pJ':          e_fine_encoder * 1e12,
        'e_transformer_pJ':           e_transformer * 1e12,
        'e_boundary_pJ':              e_boundary * 1e12,
        'e_lif_head_pJ':              e_lif_snn * 1e12,
        'e_seg_head_pJ':              e_seg_head * 1e12,
        'e_asp_total_pJ':             e_asp_total * 1e12,
        'e_ann_equiv_pJ':             e_ann_equiv * 1e12,
        'alpha_system':               alpha_system,
        'alpha_head':                 alpha_head,
        'alpha_encoder':              alpha_encoder,
        'alpha_seg_head':             alpha_seg_head,
        'spiking_frac_of_compute':    spiking_frac,
    }


def print_energy_report(energy: dict):
    """Pretty-print the energy accounting."""
    print(f"\n{'='*56}")
    print(f"  Energy Accounting (per sample)")
    print(f"{'='*56}")
    print(f"  Encoder type           : {energy['encoder_type']}"
          f"  (T_enc={energy['encoder_T']})")
    print(f"  Seg-head type          : {energy['seg_head_type']}"
          f"  (T_seg={energy['seg_head_T']})")
    print(f"  Head firing rate       : {energy['mean_firing_rate_head']*100:.2f}%")
    if energy['mean_firing_rate_encoder'] is not None:
        print(f"  Encoder firing rate    : {energy['mean_firing_rate_encoder']*100:.2f}%")
    if energy['mean_firing_rate_seg_head'] is not None:
        print(f"  Seg-head firing rate   : {energy['mean_firing_rate_seg_head']*100:.2f}%")
    print(f"  {'-'*52}")
    print(f"  Coarse encoder         : {energy['e_encoder_pJ']:>14.1f} pJ")
    if energy['e_fine_encoder_pJ'] > 0:
        print(f"  Fine encoder (analog)  : {energy['e_fine_encoder_pJ']:>14.1f} pJ")
    print(f"  Transformer (analog)   : {energy['e_transformer_pJ']:>14.1f} pJ")
    if energy['e_boundary_pJ'] > 0:
        print(f"  Boundary head (analog) : {energy['e_boundary_pJ']:>14.1f} pJ")
    print(f"  LIF head               : {energy['e_lif_head_pJ']:>14.1f} pJ")
    print(f"  Seg head               : {energy['e_seg_head_pJ']:>14.1f} pJ")
    print(f"  {'-'*52}")
    print(f"  ASP-SNN total          : {energy['e_asp_total_pJ']:>14.1f} pJ")
    print(f"  ANN-equivalent total   : {energy['e_ann_equiv_pJ']:>14.1f} pJ")
    print(f"  {'-'*52}")
    print(f"  System energy ratio α  : {energy['alpha_system']:>7.2f}x")
    print(f"  Encoder-only ratio     : {energy['alpha_encoder']:>7.2f}x")
    print(f"  Seg-head-only ratio    : {energy['alpha_seg_head']:>7.2f}x")
    print(f"  LIF-head-only ratio    : {energy['alpha_head']:>7.2f}x")
    print(f"  Spiking % of compute   : {energy['spiking_frac_of_compute']*100:.2f}%")
    print(f"{'='*56}")
    if energy['encoder_type'] == 'analog' and energy['spiking_frac_of_compute'] < 0.05:
        print("  NOTE: analog encoder dominates. System-level savings are")
        print("  limited until encoder is spiked (set encoder_type=spiking).")
        print(f"{'='*56}")