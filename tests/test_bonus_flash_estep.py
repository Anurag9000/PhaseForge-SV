"""Flash GMM research comparator does not fabricate GPU values on CPU CI."""
import pytest
from phaseforge.bonus_flash_estep import benchmark_case


def test_ibm_flash_cpu_absence_is_clearly_marked(tmp_path):
    import torch
    if torch.cuda.is_available():
        pytest.skip("This test checks CPU-only availability reporting")
    result = benchmark_case(
        n=120,k=3,d=4,output=tmp_path,repeats=1,
        allow_missing_flash=True
    )
    assert result["measurements"]["native_chunked"]["status"] == "complete"
    assert result["measurements"]["ibm_flash"]["status"] == "skipped"
    assert result["scope"] == "one E-step only, not GMM fit, MAP or speaker EER"
    assert (tmp_path/"flash_estep_comparison.json").exists()


def test_missing_gpu_without_optional_skip_is_error(tmp_path):
    import torch
    if torch.cuda.is_available():
        pytest.skip("CPU-only guard")
    with pytest.raises(RuntimeError, match="CUDA GPU"):
        benchmark_case(
            n=120,k=3,d=4,output=tmp_path,
            allow_missing_flash=False
        )
