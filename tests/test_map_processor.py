import numpy as np

from dipoleska.utils.map_process import MapProcessor


def test_map_processor_accepts_multiple_maps() -> None:
    base_map = np.arange(12, dtype=int)
    maps = [base_map, base_map + 10]

    processor = MapProcessor(maps)
    processor.mask(
        output_frame='C',
        classification=['north_equatorial'],
        radius=[30],
    )

    masked = processor.density_maps
    assert len(masked) == 2
    nan_mask = np.isnan(masked[0])
    assert np.array_equal(nan_mask, np.isnan(masked[1]))
    np.testing.assert_allclose(
        masked[0][~nan_mask],
        masked[1][~nan_mask] - 10,
    )
