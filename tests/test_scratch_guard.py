"""Disk-safety: MaxDisk sizing from physical free space + route injection."""

from ivette.util import hardware
from ivette.module import gaussian16_core as g16


def test_recommend_max_disk_scales_with_free_and_jobs():
    # 40 GB free, 2 jobs, 50% fraction → 10 GB per job (2×10=20 < 40, headroom kept).
    assert hardware.recommend_max_disk_gb(2, free_mb=40 * 1024) == 10
    assert hardware.recommend_max_disk_gb(1, free_mb=40 * 1024) == 20
    # Never returns below the floor even on a nearly-full disk.
    assert hardware.recommend_max_disk_gb(4, free_mb=1024, floor_gb=2) == 2
    # No reading available → floor.
    assert hardware.recommend_max_disk_gb(2, free_mb=0) == 2


def test_build_gjf_injects_maxdisk_into_route():
    gjf = g16.build_gjf("  C 0 0 0", "x.chk", method="PBE0", basis_set="6-311G",
                        operation="opt", cosmo=True, max_disk="8GB")
    assert "MaxDisk=8GB" in gjf
    # order: operation, solvent, then MaxDisk
    assert "opt scrf=(cpcm,solvent=water) MaxDisk=8GB" in gjf


def test_build_gjf_no_maxdisk_when_unset():
    gjf = g16.build_gjf("  C 0 0 0", "x.chk", operation="sp")
    assert "MaxDisk" not in gjf


def test_physical_free_mb_returns_a_number():
    # Can't assert an exact value, but it must return a positive int (or None only
    # if every candidate path failed, which shouldn't happen on a real FS).
    v = hardware.physical_free_mb()
    assert v is None or (isinstance(v, int) and v > 0)
