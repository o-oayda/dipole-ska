# Map Collection Loading

This project supports two ways to load SKA map collections:

1. **Legacy (template-based)**: you provide the same parameters as before (`snr_cut`, `lower_flux_limit`, `lower_z_limit`, `gal_cut`, `nside`, `map_types`, and doppler/newsize/input flags). The loader builds the exact expected filenames from those parameters and loads them. Behavior is unchanged from the original API (note however the boolean kwargs included at the top of the MapCollectionLoader call below).
2. **Discovery (filename-parsing)**: the loader scans the mapcollection directories on disk, parses attributes from filenames, and groups files that share the same attribute set. This automatically supports new attributes in filenames without code changes.

You can also pass `use_base_rms=True` to swap the per-map RMS files with a reference RMS map (per `nside`) distributed alongside the data.

## Legacy usage (unchanged)

```python
from dipoleska.utils.map_read import MapCollectionLoader

loader = MapCollectionLoader(
    use_doppler=True,
    use_newsizes=False,
    use_input_variant=False,
    nside=64,
    snr_cut=10,
    lower_flux_limit='1e-4',
    lower_z_limit='0.5',
    gal_cut=10,
    map_types=['counts', 'rms'],  # or ['all']
)
maps = loader.map_collections  # dict[map_type -> np.ndarray]
```

## Discovery usage (new)

The loader will default to discovery mode if you omit legacy params or if you explicitly pass `base_dirs`. By default, it scans:

- `data/ska/mapcollections`
- `data/ska/mapcollections_input`

You can override `base_dirs` to point to other roots (e.g., temporary test data or additional delivery locations).

### Inspect what’s available

```python
loader = MapCollectionLoader()  # discovery mode by default
entries = loader.list_available()  # grouped by shared attributes
for entry in entries:
    print(loader.describe(entry))
```

Each grouped entry has:
```python
{
  "attrs": {"doppler": True, "newsizes": False, "input_variant": False,
            "nside": 64, "flux": 1e-4, "snr": 5,
            "z": 0.3, "z_upper": 5.0, "gal": 10.0, ...},
  "files": {
      "counts": {"path": "...countmap_...fits"},
      "rms":    {"path": "...rmsmap_...fits"},
      # ... other map types present for that attribute set ...
  }
}
```

### Load maps

Use `load` to populate the cache, then read via `map_collections`.
If you don't pass any filters to load, beware that you will load all the available maps, which is slower and chews through memory.

```python
# Load everything discovered
loader.load()
maps = loader.map_collections

# Load only a specific attribute set (e.g., doppler + nside64 + snr5)
loader.clear_cache()
loader.load(filter_attrs={"doppler": True, "nside": 64, "snr": 5})
maps = loader.map_collections

# Load only certain map types within that filter
loader.clear_cache()
loader.load(filter_attrs={"doppler": True, "nside": 64}, map_types=["counts", "rms"])
maps = loader.map_collections
```

### What does `map_collections` return?

- If a single attribute set is loaded and there is only one file per map type, you get a simple `dict[map_type -> data]` (like the legacy output).
- If multiple attribute sets are loaded or duplicates per map type exist, you get a `list` of grouped entries, each shaped like the `entries` shown above but with `data` included:

```python
{
  "attrs": {...},
  "files": {
    "counts": {"path": "...", "data": <np.ndarray>},
    "rms": {"path": "...", "data": <np.ndarray>},
    # ...
  }
}
```

### Refreshing / reloading

- `load` overwrites the in-memory cache each time.
- `map_collections` caches the last load; to force reload/rescan, call `loader.clear_cache(include_discovery=True)` and then `loader.map_collections` or `loader.load(...)` again.

## Attribute parsing notes

- Doppler/newsizes/input_variant are inferred from directory names (`doppler`, `no_doppler`, `new_sizes`/`newsizes`, `mapcollections_input`).
- Redshift tokens like `_z0.3_z5.0` are parsed into `z` (lower) and `z_upper` (upper).
- Unknown tokens are retained as-is in `attrs`, so new attributes in filenames are automatically available for filtering.
