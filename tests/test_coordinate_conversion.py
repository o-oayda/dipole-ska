import numpy as np
from astropy.coordinates import Galactic, ICRS, SkyCoord

from dipoleska.utils.constants import CMB_L, CMB_B, CMB_RA, CMB_DEC
from dipoleska.utils.physics import change_source_coordinates


def _sample_gaussian_coords(
        center_lon: float,
        center_lat: float,
        sigma_lon: float,
        sigma_lat: float,
        size: int,
        seed: int
    ) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed=seed)
    longitude = np.mod(rng.normal(center_lon, sigma_lon, size=size), 360.0)
    latitude = np.clip(rng.normal(center_lat, sigma_lat, size=size), -90.0, 90.0)
    return longitude, latitude


def _assert_circular_allclose(
        actual: np.ndarray,
        expected: np.ndarray,
        atol: float = 1e-9
    ) -> None:
    delta = (np.asarray(actual) - np.asarray(expected) + 180.0) % 360.0 - 180.0
    np.testing.assert_allclose(delta, np.zeros_like(delta), atol=atol)


def test_cmb_gal2eq_matches_constants():
    galactic_longitude = np.array([CMB_L], dtype=np.float64)
    galactic_latitude = np.array([CMB_B], dtype=np.float64)

    ra_deg, dec_deg = change_source_coordinates(
        galactic_longitude,
        galactic_latitude,
        native_coordinates='galactic',
        target_coordinates='equatorial'
    )

    np.testing.assert_allclose(ra_deg, [CMB_RA], atol=1e-4)
    np.testing.assert_allclose(dec_deg, [CMB_DEC], atol=1e-4)


def test_gal2eq_uniform_distribution_matches_astropy():
    rng = np.random.default_rng(seed=12345)
    galactic_longitude = rng.uniform(0.0, 360.0, size=256)
    galactic_latitude = rng.uniform(-90.0, 90.0, size=256)

    transformed_ra, transformed_dec = change_source_coordinates(
        galactic_longitude,
        galactic_latitude,
        native_coordinates='galactic',
        target_coordinates='equatorial'
    )

    expected = SkyCoord(
        galactic_longitude,
        galactic_latitude,
        frame=Galactic,
        unit='deg'
    ).transform_to(ICRS())

    np.testing.assert_allclose(
        transformed_ra,
        expected.ra.value,
        atol=1e-9
    )
    np.testing.assert_allclose(
        transformed_dec,
        expected.dec.value,
        atol=1e-9
    )


def test_gal2eq_gaussian_distribution_matches_astropy():
    galactic_longitude, galactic_latitude = _sample_gaussian_coords(
        center_lon=CMB_L,
        center_lat=CMB_B,
        sigma_lon=2.0,
        sigma_lat=2.0,
        size=128,
        seed=67890
    )

    transformed_ra, transformed_dec = change_source_coordinates(
        galactic_longitude,
        galactic_latitude,
        native_coordinates='galactic',
        target_coordinates='equatorial'
    )

    expected = SkyCoord(
        galactic_longitude,
        galactic_latitude,
        frame=Galactic,
        unit='deg'
    ).transform_to(ICRS())

    np.testing.assert_allclose(
        transformed_ra,
        expected.ra.value,
        atol=1e-9
    )
    np.testing.assert_allclose(
        transformed_dec,
        expected.dec.value,
        atol=1e-9
    )


def test_cmb_eq2gal_matches_constants():
    equatorial_ra = np.array([CMB_RA], dtype=np.float64)
    equatorial_dec = np.array([CMB_DEC], dtype=np.float64)

    lon_deg, lat_deg = change_source_coordinates(
        equatorial_ra,
        equatorial_dec,
        native_coordinates='equatorial',
        target_coordinates='galactic'
    )

    np.testing.assert_allclose(lon_deg, [CMB_L], atol=1e-4)
    np.testing.assert_allclose(lat_deg, [CMB_B], atol=1e-4)


def test_eq2gal_uniform_distribution_matches_astropy():
    rng = np.random.default_rng(seed=24680)
    equatorial_ra = rng.uniform(0.0, 360.0, size=256)
    equatorial_dec = rng.uniform(-90.0, 90.0, size=256)

    lon_deg, lat_deg = change_source_coordinates(
        equatorial_ra,
        equatorial_dec,
        native_coordinates='equatorial',
        target_coordinates='galactic'
    )

    expected = SkyCoord(
        equatorial_ra,
        equatorial_dec,
        frame=ICRS,
        unit='deg'
    ).transform_to(Galactic())

    np.testing.assert_allclose(lon_deg, expected.l.value, atol=1e-9)
    np.testing.assert_allclose(lat_deg, expected.b.value, atol=1e-9)


def test_eq2gal_gaussian_distribution_matches_astropy():
    equatorial_ra, equatorial_dec = _sample_gaussian_coords(
        center_lon=CMB_RA,
        center_lat=CMB_DEC,
        sigma_lon=2.0,
        sigma_lat=2.0,
        size=128,
        seed=13579
    )

    lon_deg, lat_deg = change_source_coordinates(
        equatorial_ra,
        equatorial_dec,
        native_coordinates='equatorial',
        target_coordinates='galactic'
    )

    expected = SkyCoord(
        equatorial_ra,
        equatorial_dec,
        frame=ICRS,
        unit='deg'
    ).transform_to(Galactic())

    np.testing.assert_allclose(lon_deg, expected.l.value, atol=1e-9)
    np.testing.assert_allclose(lat_deg, expected.b.value, atol=1e-9)


def test_eq_gal_ecl_eq_round_trip_preserves_coordinates():
    rng = np.random.default_rng(seed=424242)
    equatorial_ra = rng.uniform(0.0, 360.0, size=256)
    equatorial_dec = rng.uniform(-70.0, 70.0, size=256)

    gal_lon, gal_lat = change_source_coordinates(
        equatorial_ra,
        equatorial_dec,
        native_coordinates='equatorial',
        target_coordinates='galactic'
    )
    ecl_lon, ecl_lat = change_source_coordinates(
        gal_lon,
        gal_lat,
        native_coordinates='galactic',
        target_coordinates='ecliptic'
    )
    round_ra, round_dec = change_source_coordinates(
        ecl_lon,
        ecl_lat,
        native_coordinates='ecliptic',
        target_coordinates='equatorial'
    )

    _assert_circular_allclose(round_ra, equatorial_ra, atol=1e-9)
    np.testing.assert_allclose(round_dec, equatorial_dec, atol=1e-9)
