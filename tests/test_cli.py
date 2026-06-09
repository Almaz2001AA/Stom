from stomcore.cli import main
from stomcore.nifti_io import load_volume_nifti


def test_cli_converts_dicom_to_nifti(dicom_series, tmp_path, capsys):
    out = tmp_path / "out.nii.gz"
    code = main([dicom_series, str(out)])
    assert code == 0
    assert out.exists()
    vol = load_volume_nifti(out)
    assert vol.shape == (8, 16, 16)
    assert "out.nii.gz" in capsys.readouterr().out


def test_cli_reports_error_on_empty_dir(tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    out = tmp_path / "out.nii.gz"
    code = main([str(empty), str(out)])
    assert code == 1
    assert not out.exists()
    assert "error" in capsys.readouterr().err.lower()
