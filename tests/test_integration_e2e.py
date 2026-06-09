import io

from stomcore.mask_io import load_mask_nifti
from stomcore.nifti_io import load_volume_nifti
from stomserver.segmentation.runner import FakeRunner
from stomserver.segmentation.worker import _run_segmentation


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def test_full_pipeline(client, db_factory, storage, account_token, nifti_bytes, tmp_path):
    _, token = account_token

    # Wire the queue to run the worker synchronously on enqueue.
    client.app.state.queue.set_sync_handler(
        lambda job_id: _run_segmentation(job_id, db_factory, storage, FakeRunner())
    )

    study_id = client.post("/studies", files={"file": ("v.nii.gz", nifti_bytes)},
                           headers=_auth(token)).json()["study_id"]
    job = client.post(f"/studies/{study_id}/segment", headers=_auth(token)).json()
    assert client.get(f"/jobs/{job['job_id']}", headers=_auth(token)).json()["status"] == "done"

    mask_resp = client.get(f"/studies/{study_id}/masks", headers=_auth(token))
    labels_resp = client.get(f"/studies/{study_id}/masks/labels", headers=_auth(token))
    assert mask_resp.status_code == 200 and labels_resp.status_code == 200

    # Persist returned files and load them back; geometry must match the uploaded volume.
    (tmp_path / "m.nii.gz").write_bytes(mask_resp.content)
    (tmp_path / "m.json").write_bytes(labels_resp.content)
    mask = load_mask_nifti(tmp_path / "m.nii.gz", tmp_path / "m.json")

    (tmp_path / "v.nii.gz").write_bytes(nifti_bytes)
    volume = load_volume_nifti(tmp_path / "v.nii.gz")
    assert mask.is_compatible_with(volume)
    assert mask.shape == volume.shape
