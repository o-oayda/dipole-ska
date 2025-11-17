import numpy as np
import pytest

from dipoleska.utils.map_read import MapCollectionLoader


def test_discovery_parses_attributes(tmp_path):
    base = tmp_path / "data" / "ska" / "mapcollections" / "doppler_new_sizes"
    base.mkdir(parents=True)
    file_path = base / "countmap_nside64_flux1e-4_snr5_newattrX.txt"
    file_path.write_text("1 2\n3 4\n")

    loader = MapCollectionLoader(base_dirs=[
        str(tmp_path / "data" / "ska" / "mapcollections"),
        str(tmp_path / "data" / "ska" / "mapcollections_input")
    ])
    entries = loader.list_available()

    assert len(entries) == 1
    entry = entries[0]
    attrs = entry["attrs"]
    assert attrs["nside"] == 64
    assert pytest.approx(attrs["flux"]) == 1e-4
    assert attrs["snr"] == 5
    assert attrs["newattr"] == "X"
    assert attrs["doppler"] is True
    assert attrs["newsizes"] is True
    assert attrs["input_variant"] is False
    assert set(entry["files"].keys()) == {"counts"}


def test_load_filters_and_reads_matching_file(tmp_path):
    base = tmp_path / "data" / "ska" / "mapcollections" / "no_doppler"
    base.mkdir(parents=True)
    file_a = base / "countmap_nside64_flux1e-4_snr5.txt"
    file_b = base / "countmap_nside128_flux1e-4_snr5.txt"
    file_a.write_text("10 20\n30 40\n")
    file_b.write_text("1 2\n3 4\n")

    loader = MapCollectionLoader(base_dirs=[str(tmp_path / "data" / "ska" / "mapcollections")])
    loader.load(filter_attrs={"nside": 64, "doppler": False})
    collections = loader.map_collections

    assert "counts" in collections
    loaded = collections["counts"]
    np.testing.assert_array_equal(loaded, np.array([[10, 20], [30, 40]]))


def test_map_collections_property_uses_discovery(tmp_path):
    base = tmp_path / "data" / "ska" / "mapcollections" / "doppler"
    base.mkdir(parents=True)
    file_path = base / "rmsmap_nside64_flux1e-4_snr10.txt"
    file_path.write_text("5 6\n7 8\n")

    loader = MapCollectionLoader(base_dirs=[str(tmp_path / "data" / "ska" / "mapcollections")])
    result = loader.map_collections

    assert "rms" in result
    np.testing.assert_array_equal(result["rms"], np.array([[5, 6], [7, 8]]))


def test_grouped_entries_bundle_multiple_maps(tmp_path):
    base = tmp_path / "data" / "ska" / "mapcollections" / "doppler_newsizes"
    base.mkdir(parents=True)
    (base / "countmap_nside64_flux1e-4_snr5_z0.3_z5.0.txt").write_text("1 2\n")
    (base / "rmsmap_nside64_flux1e-4_snr5_z0.3_z5.0.txt").write_text("3 4\n")

    loader = MapCollectionLoader(base_dirs=[str(tmp_path / "data" / "ska" / "mapcollections")])
    entries = loader.list_available()

    assert len(entries) == 1
    entry = entries[0]
    assert set(entry["files"].keys()) == {"counts", "rms"}
    assert entry["attrs"]["doppler"] is True
    assert entry["attrs"]["newsizes"] is True
    assert entry["attrs"]["z"] == 0.3
    assert entry["attrs"]["z_upper"] == 5.0
    assert entry["id"].startswith("mapcollections-doppler-newsizes-nside64_flux1e-4_snr5_z0.3_z5.0")


def test_clear_cache_allows_reload(tmp_path):
    base = tmp_path / "data" / "ska" / "mapcollections" / "doppler"
    base.mkdir(parents=True)
    target = base / "countmap_nside64_flux1e-4_snr5_z0.0_z5.0.txt"
    target.write_text("1 2\n3 4\n")

    loader = MapCollectionLoader(base_dirs=[str(tmp_path / "data" / "ska" / "mapcollections")])
    first = loader.map_collections
    np.testing.assert_array_equal(first["counts"], np.array([[1, 2], [3, 4]]))

    target.write_text("5 6\n7 8\n")
    loader.clear_cache(include_discovery=True)
    second = loader.map_collections
    np.testing.assert_array_equal(second["counts"], np.array([[5, 6], [7, 8]]))
