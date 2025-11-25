from typing import Any, List, Literal
from pathlib import Path
import healpy as hp
from numpy.typing import NDArray
import numpy as np


class MapLoader:
    def __init__(self,
            briggs_weighting: Literal[-1, 0, 1],
            configuration: Literal['AA', 'AA4']
    ):
        '''
        Class for loading in SKA maps. Call the load method to return a map.

        :param briggs_weighting: Briggs weighting used to generate the map.
        :param configuration: SKA telescope configuration used to 
                            generate the map.
        '''
        if briggs_weighting == -1:
            self.briggs_weighting = 'n1'
        else:
            self.briggs_weighting = str(briggs_weighting)
        
        self.configuration = configuration

    def load(self,
            map_number: int
    ) -> NDArray[np.int_]:
        '''
        Load and return SKA fits density map based on congigured settings
        and specified map number.

        :param map_number: Number appearing in fits file, e.g. `map_1.fits`
            refers to map 1.
        :return: Healpy density map.
        '''
        self.map_number = map_number
        self.file_path = (
            f'data/ska/briggs_{self.briggs_weighting}/{self.configuration}'
        )
        self.file_name = f'map_{self.map_number}.fits'
        try:
            print(f'Reading in {self.file_path}/{self.file_name}...')
            self.density_map = hp.read_map(
                f'{self.file_path}/{self.file_name}',
                nest=False
            )
            return self.density_map
        except FileNotFoundError as e:
            raise Exception(f'''
                            Cannot find file. File details are as follows:
                            Briggs weighting: {self.briggs_weighting}
                            Configuration: {self.configuration}
                            Map number: {self.map_number}
                            Path: {self.file_path}/{self.file_name}
                            ''') from e

class MapCollectionLoader:
    """
    Load SKA map collections either via legacy filename templates or by
    discovering files on disk and parsing attributes from filenames.
    
    - Legacy mode (no base_dirs, all legacy params provided) builds the exact
      expected file paths and loads those maps.
    - Discovery mode scans mapcollection directories, parses attributes (including
      new/unknown tokens), and groups files that share attribute sets. Use
      `load(filter_attrs=..., map_types=...)` to select which files to read, then
      access results via `map_collections`. Call `clear_cache()` to release loaded
      data and optionally refresh discovery. Discovery defaults to scanning
      `data/ska/mapcollections` and `data/ska/mapcollections_input`, unless
      `base_dirs` is overridden.
    """
    def __init__(self,
             snr_cut: Literal[5, 10, 15] | None = None,
             lower_flux_limit: Literal['5e-5', '1e-4', '5e-4', '1e-3'] | str | None = None,
             lower_z_limit: Literal['0', '0.3', '0.5', '1.2'] | str | None = None,
             gal_cut: Literal[0,5,10] | float | None = None,
             map_types: List[
                 Literal[
                     'counts', 'rms', 'alpha', 'redshift', 
                     'flux','info','all'
                 ]
             ] | None = None,
             nside: Literal[64, 256] | int | None = 64,
             use_doppler: bool = True,
             use_newsizes: bool = False,
             use_input_variant: bool = False,
             use_base_rms: bool = False,
             base_dirs: List[str] | None = None,
        ) -> None:
        '''
        Loader for SKA map collections with two modes:
        - Legacy: uses fixed filename templates (requires legacy params).
        - Discovery: scans mapcollection directories, parses filenames into
          attributes (auto-supports new attributes), and loads matching maps.
        
        :param use_doppler: Use the Doppler-boosted or non-Doppler-boosted
            (rest frame) samples.
        :param use_newsizes: Use the updated simulations with the AGN size error
            fixed.
        :param use_input_variant: Use the sims from `SimulationInput`; these
            do not have observational effects included
        :param use_base_rms: Replace the RMS maps with a reference baseline map
            (per nside) instead of the RMS map bundled with each collection.
        :param snr_cut: SNR cut to apply when loading maps.
        :param lower_flux_limit: Lower flux limit to apply when loading maps.
        :param lower_z_limit: Lower redshift limit to apply when loading maps.
        :param gal_cut: Galactic cut to apply when loading maps.
        :param nside: Healpy nside parameter for the maps.
        :param map_types: Specify which maps to load in the collections dict.
            If 'all' is passed in the list, the final map_collections dict
            will contain all available data types.
        :param base_dirs: When provided, maps are discovered dynamically from
            these directories by parsing filenames for attributes. This allows
            new attributes to be supported without code changes.
        :param use_base_rms: Substitute the per-collection RMS maps with the
            reference RMS maps (see `MapCollectionLoader._base_rms_maps`).
        '''
        
        # Cache can hold either a dict[map_type, data] (legacy or single-group)
        # or a list of grouped entries when multiple attribute sets are loaded.
        self._map_collections: dict[str, Any] | list[dict[str, Any]] = {}
        self.snr_cut = snr_cut
        self.lower_flux_limit = float(lower_flux_limit) if lower_flux_limit is not None else None
        self.lower_z_limit = float(lower_z_limit) if lower_z_limit is not None else None
        self.gal_cut = gal_cut
        self.nside = nside
        self.default_upper_z_limit = float("5.0")
        self.newsizes_upper_z_limit = float("10.0")

        self._base_rms_maps = {
            64: 'mapcollections/doppler/rmsmap_nside64_flux1e-05_snr5_z0.0_z5.0_gal0.0.fits',
            256: 'mapcollections/doppler/rmsmap_nside256_flux1e-05_snr5_z0.0_z5.0_gal0.0.fits'
        }
        self._base_rms_cache: dict[int, Any] = {}

        if use_newsizes:
            newsize_str = '_new_sizes'
            self.upper_z_limit = self.newsizes_upper_z_limit
        else:
            newsize_str = ''
            self.upper_z_limit = self.default_upper_z_limit

        if use_input_variant:
            input_str = '_input'
        else:
            input_str = ''

        self.use_doppler = use_doppler
        self.use_newsizes = use_newsizes
        self.use_input_variant = use_input_variant
        self.use_base_rms = use_base_rms

        legacy_candidate = base_dirs is None and map_types is not None
        missing_legacy = []
        if legacy_candidate:
            required = {
                "snr_cut": self.snr_cut,
                "lower_flux_limit": self.lower_flux_limit,
                "lower_z_limit": self.lower_z_limit,
                "gal_cut": self.gal_cut,
                "nside": self.nside,
            }
            missing_legacy = [name for name, value in required.items() if value is None]
            if missing_legacy:
                raise ValueError(
                    "Missing required parameters for legacy mode: "
                    + ", ".join(missing_legacy)
                    + ". Provide base_dirs to use discovery mode instead."
                )

        legacy_requested = legacy_candidate and not missing_legacy

        if legacy_requested:
            self.base_dirs = None
        elif base_dirs is None:
            # Default discovery roots; searched recursively.
            self.base_dirs = [
                "data/ska/mapcollections_24-11-25",
                "data/ska/mapcollections_input_24-11-25"
            ]
        else:
            self.base_dirs = base_dirs

        if self.base_dirs:
            self.path_to_files = None
        elif use_doppler:
            self.path_to_files = f'data/ska/mapcollections{input_str}/doppler{newsize_str}/'
        else:
            self.path_to_files = f'data/ska/mapcollections{input_str}/no_doppler{newsize_str}/'

        if self.base_dirs:
            self.file_configuration = ""
        # since the input_variant files have no snr attribute, drop it from the filename
        elif use_input_variant:
            self.file_configuration = (
                f'_nside{self.nside}_flux{self.lower_flux_limit}_snr0'
                f'_z{self.lower_z_limit}_z{self.upper_z_limit}_gal{self.gal_cut}.0'
            )
        else:
            self.file_configuration = (
                f'_nside{self.nside}_flux{self.lower_flux_limit}_snr{self.snr_cut}'
                f'_z{self.lower_z_limit}_z{self.upper_z_limit}_gal{self.gal_cut}.0'
            )
        self.map_dict = {
            'counts': ('countmap', '.fits'),
            'rms': ('rmsmap', '.fits'),
            'alpha': ('alphamap', '.fits'),
            'redshift': ('zhist', '.txt'),
            'flux': ('fluxhist', '.txt'),
            'info': ('xa', '.txt')
        }
        if map_types is None:
            self.map_types = list(self.map_dict.keys())
        elif 'all' in map_types:
            self.map_types = self.map_dict.keys()
        else:
            self.map_types = map_types

        self._available_cache: List[dict[str, Any]] | None = None

    def get_file_attributes(self) -> str:
        map_types_str = ", ".join(sorted(self.map_types))
        if self.base_dirs:
            return "\n".join([
                f"Base directories: {', '.join(self.base_dirs)}",
                f"Map types: {map_types_str}",
                "Attributes are discovered from filenames.",
                f"Using base RMS map: {self.use_base_rms}",
            ])

        return "\n".join([
            f"Base path: {self.path_to_files}",
            f"File stem: {self.file_configuration[1:]}",  # remove leading underscore
            f"Map types: {map_types_str}",
            f"Nside: {self.nside}",
            f"SNR cut: {self.snr_cut}",
            f"Lower flux limit: {self.lower_flux_limit}",
            f"Redshift range: {self.lower_z_limit} - {self.upper_z_limit}",
            f"Galactic cut: {self.gal_cut}",
            f"Using Doppler boost: {self.use_doppler}",
            f"Using new sizes: {self.use_newsizes}",
            f"Using input variant: {self.use_input_variant}",
            f"Using base RMS map: {self.use_base_rms}"
        ])

    def list_available(self, map_type: str | None = None, refresh: bool = False,
                       grouped: bool = True) -> List[dict[str, Any]]:
        '''
        Discover and return all available map files in the configured base
        directories. Filenames are parsed into a dictionary of attributes.
        When grouped=True (default), files with identical attributes are
        bundled together under a single entry with a 'files' mapping.
        When grouped=False, the raw per-file descriptors are returned.
        '''
        if not self.base_dirs:
            raise ValueError("list_available is only supported when base_dirs are provided.")

        entries = self._discover_files(refresh=refresh)
        if not grouped:
            if map_type:
                entries = [entry for entry in entries if entry["map_type"] == map_type]
            return entries

        grouped_entries = self._group_entries(entries)
        if map_type:
            grouped_entries = [
                entry for entry in grouped_entries if map_type in entry["files"]
            ]
        return grouped_entries

    def describe(self, entry: dict[str, Any]) -> str:
        '''
        Render a human-readable description of a discovered map entry.
        '''
        attrs = ", ".join(f"{k}={v}" for k, v in sorted(entry.get("attrs", {}).items()))
        if "files" in entry:
            maps = ", ".join(sorted(entry["files"].keys()))
            paths = "; ".join(f"{mt}:{info['path']}" for mt, info in entry["files"].items())
            return f"maps=[{maps}] attrs=({attrs}) paths=[{paths}]"
        return f"{entry['map_type']} ({entry['ext']}): {attrs} @ {entry['path']}"

    def _discover_files(self, refresh: bool = False) -> List[dict[str, Any]]:
        from pathlib import Path
        if self._available_cache is not None and not refresh:
            return self._available_cache

        entries: List[dict[str, Any]] = []
        for base in self.base_dirs:
            base_path = Path(base)
            if not base_path.exists():
                continue
            for path in base_path.glob("**/*"):
                if not path.is_file():
                    continue
                if path.suffix not in {".fits", ".txt"}:
                    continue
                descriptor = self._parse_descriptor(path)
                entries.append(descriptor)
        self._available_cache = entries
        return entries

    def _group_entries(self, entries: List[dict[str, Any]], include_data: bool = False) -> List[dict[str, Any]]:
        grouped: dict[tuple[tuple, ...], dict[str, Any]] = {}
        for entry in entries:
            key = tuple(sorted(entry["attrs"].items()))
            if key not in grouped:
                identifier = self._build_identifier(entry)
                grouped[key] = {
                    "attrs": entry["attrs"],
                    "files": {},
                    "id": identifier
                }
            file_info = {"path": entry["path"]}
            if include_data and "data" in entry:
                file_info["data"] = entry["data"]
            grouped[key]["files"][entry["map_type"]] = file_info
        return list(grouped.values())

    def _parse_descriptor(self, path) -> dict[str, Any]:
        '''
        Parse a filename into a descriptor that includes map_type, attributes,
        extension, and the absolute path. Unknown tokens are preserved as-is.
        '''
        src_attrs = self._source_attrs(path)
        name = path.name
        stem = name.rsplit(".", 1)[0]
        tokens = stem.split("_")
        prefix = tokens[0]
        map_type = self._infer_map_type(prefix)
        attrs: dict[str, Any] = dict(src_attrs)
        collection_dir = self._collection_dir(path)
        if collection_dir:
            attrs["collection_dir"] = collection_dir
        for token in tokens[1:]:
            key, value = self._split_token(token)
            if key == "z":
                if "z" not in attrs:
                    attrs["z"] = value  # lower z
                else:
                    attrs["z_upper"] = value  # upper z
            else:
                attrs[key] = value
        return {
            "map_type": map_type,
            "ext": path.suffix,
            "attrs": attrs,
            "path": str(path)
        }

    def _build_identifier(self, entry: dict[str, Any]) -> str:
        base_parts = []
        path = Path(entry["path"])
        collection_dir = entry["attrs"].get("collection_dir")
        if not collection_dir:
            collection_dir = self._collection_dir(path) or path.parent.name or "external"
        base_parts.append(collection_dir)

        base_parts.append("doppler" if entry["attrs"].get("doppler") else "no_doppler")
        if entry["attrs"].get("newsizes"):
            base_parts.append("newsizes")
        # Build from filename tokens (excluding map_type and ext)
        stem_tokens = path.name.rsplit(".", 1)[0].split("_")
        if stem_tokens:
            stem_tokens = stem_tokens[1:]  # drop leading map_type prefix
        base_parts.append("_".join(stem_tokens))
        return "-".join(part for part in base_parts if part)

    def _collection_dir(self, path: Path) -> str | None:
        parts = path.parts
        for idx in range(len(parts) - 1):
            if parts[idx] == "data" and parts[idx + 1] == "ska":
                if idx + 2 < len(parts):
                    return parts[idx + 2]
                break
        return None

    def _source_attrs(self, path) -> dict[str, Any]:
        '''
        Infer doppler/newsizes/input_variant from the directory structure.
        '''
        parts = [p.lower() for p in path.parts]
        def part_contains(token: str) -> bool:
            return any(token in p for p in parts)

        input_variant = part_contains("mapcollections_input")
        no_doppler = part_contains("no_doppler")
        doppler = part_contains("doppler") and not no_doppler
        newsizes = part_contains("new_sizes") or part_contains("newsizes")
        return {
            "doppler": doppler,
            "newsizes": newsizes,
            "input_variant": input_variant
        }

    def _split_token(self, token: str) -> tuple[str, Any]:
        import re
        match = re.match(r"([A-Za-z]+)(.*)", token)
        if not match:
            return token, None
        key, raw_value = match.groups()
        # If no trailing value, try to split on a lower-to-upper boundary
        if raw_value == "":
            boundary = re.match(r"([a-z]+)([A-Z].*)", token)
            if boundary:
                key, raw_value = boundary.groups()
        value = self._coerce_value(raw_value)
        return key.lower(), value

    def _coerce_value(self, raw: str) -> Any:
        if raw == "":
            return None
        try:
            # Attempt numeric conversion
            if raw.lower() in {"nan", "inf", "-inf"}:
                return raw
            if "." in raw or "e" in raw.lower():
                return float(raw)
            return int(raw)
        except ValueError:
            return raw

    def _resolve_base_rms_path(self, nside: int) -> Path:
        """
        Resolve the path to the reference RMS map for the given nside.
        Accepts absolute paths in _base_rms_maps; otherwise prefixes with
        data/ska and appends .fits when no extension is present.
        """
        if nside is None:
            raise ValueError("nside must be provided to select a base RMS map.")
        if nside not in self._base_rms_maps:
            raise ValueError(f"No base RMS map configured for nside {nside}.")
        raw_path = self._base_rms_maps[nside]
        path = Path(raw_path)
        if not path.is_absolute():
            path = Path("data/ska") / path
        if path.suffix == "":
            path = path.with_suffix(".fits")
        return path

    def _load_base_rms_map(self, nside: int) -> tuple[str, Any]:
        """
        Load the reference RMS map for the given nside, caching the result.
        """
        path = self._resolve_base_rms_path(nside)
        if nside in self._base_rms_cache:
            return str(path), self._base_rms_cache[nside]
        print(f"Reading in {path}...")
        if path.suffix == ".fits":
            data = hp.read_map(str(path), nest=False)
        else:
            data = np.loadtxt(path)
        self._base_rms_cache[nside] = data
        return str(path), data

    def _infer_map_type(self, prefix: str) -> str:
        # Try to infer from known prefixes; otherwise, use the prefix itself.
        inverse = {v[0]: k for k, v in self.map_dict.items()}
        return inverse.get(prefix, prefix)

    def _legacy_validate(self) -> None:
        if self.base_dirs:
            return
        required = {
            "snr_cut": self.snr_cut,
            "lower_flux_limit": self.lower_flux_limit,
            "lower_z_limit": self.lower_z_limit,
            "gal_cut": self.gal_cut,
            "nside": self.nside,
            "map_types": self.map_types,
        }
        missing = [name for name, value in required.items() if value is None]
        if "snr_cut" in missing and self.use_input_variant:
            missing.remove("snr_cut") # input variant maps have no snr attr
        if missing:
            raise ValueError(f"Missing required parameters for legacy loading: {', '.join(missing)}")

    def load(self, filter_attrs: dict[str, Any] | None = None,
             map_types: List[str] | None = None,
             refresh: bool = False) -> None:
        '''
        Load maps based on discovered files (if base_dirs set) or using the
        configured legacy parameters. When loading from discovered files,
        filter_attrs may be provided to select files (e.g. {"nside": 64}).
        In discovery mode, if multiple attribute sets are matched, the cache
        stores a list of grouped entries; otherwise a map_type->data dict.
        This method populates the internal cache and does not return a value;
        access results via the `map_collections` property.
        '''
        if self.base_dirs:
            entries = self.list_available(refresh=refresh, grouped=False)
            if map_types:
                entries = [e for e in entries if e["map_type"] in map_types]
            if filter_attrs:
                def _entry_matches(entry: dict[str, Any]) -> bool:
                    for key, value in filter_attrs.items():
                        if key == "id":
                            identifier = self._build_identifier(entry)
                            if identifier != value:
                                return False
                        else:
                            if entry["attrs"].get(key) != value:
                                return False
                    return True
                entries = [e for e in entries if _entry_matches(e)]
            if not entries:
                raise FileNotFoundError("No matching map files found for the given filters.")
            loaded_entries: list[dict[str, Any]] = []
            attr_keys: dict[tuple[tuple[str, Any], ...], dict[str, Any]] = {}
            attr_has_rms: set[tuple[tuple[str, Any], ...]] = set()
            for entry in entries:
                path = entry["path"]
                ext = entry["ext"]
                map_type = entry["map_type"]
                attr_key = tuple(sorted(entry["attrs"].items()))
                attr_keys.setdefault(attr_key, entry["attrs"])
                try:
                    if map_type == "rms" and self.use_base_rms:
                        nside_value = entry["attrs"].get("nside")
                        if nside_value is None:
                            raise ValueError(
                                "Cannot use base RMS map because no nside attribute "
                                "was found for this entry."
                            )
                        path, data = self._load_base_rms_map(int(nside_value))
                        attr_has_rms.add(attr_key)
                    else:
                        print(f"Reading in {path}...")
                        if ext == ".fits":
                            data = hp.read_map(path, nest=False)
                        else:
                            data = np.loadtxt(path)
                        if map_type == "rms":
                            attr_has_rms.add(attr_key)
                    loaded_entries.append({
                        "map_type": map_type,
                        "attrs": entry["attrs"],
                        "path": path,
                        "data": data,
                    })
                except FileNotFoundError as e:
                    raise FileNotFoundError(
                        f"File not found for map type '{map_type}'. Expected at: {path}"
                    ) from e
                except Exception as e:
                    raise RuntimeError(
                        f"Cannot read file for map type '{map_type}' at {path}."
                        f" Underlying error: {e}"
                    ) from e

            # Ensure every attribute group receives a base RMS entry, even if
            # no RMS file existed on disk for that collection.
            if self.use_base_rms:
                for key, attrs in attr_keys.items():
                    if key in attr_has_rms:
                        continue
                    nside_value = attrs.get("nside")
                    if nside_value is None:
                        raise ValueError(
                            "Cannot use base RMS map because no nside attribute "
                            "was found for this entry."
                        )
                    path, data = self._load_base_rms_map(int(nside_value))
                    loaded_entries.append({
                        "map_type": "rms",
                        "attrs": dict(attrs),
                        "path": path,
                        "data": data,
                    })

            grouped_loaded = self._group_entries(loaded_entries, include_data=True)
            if len(grouped_loaded) == 1 and all(len(g["files"]) == 1 for g in grouped_loaded):
                # Simple case: one attribute set, one map per type; keep legacy-ish dict
                collections = {mt: info["data"] for mt, info in grouped_loaded[0]["files"].items()}
            else:
                collections = grouped_loaded

            self._map_collections = collections
            return

        # Legacy behaviour (pre-configured filenames)
        self._legacy_validate()
        base_rms_path: str | None = None
        base_rms_data: Any = None
        if self.use_base_rms and "rms" in self.map_types:
            base_rms_path, base_rms_data = self._load_base_rms_map(int(self.nside))

        self._map_collections = {}
        for map_type in self.map_types:
            if map_type not in self.map_dict:
                raise ValueError(f"Unknown map type: {map_type}")

            base_name, ext = self.map_dict[map_type]
            file_path = (f"{self.path_to_files}{base_name}"
                        f"{self.file_configuration}{ext}")

            try:
                if map_type == "rms" and self.use_base_rms:
                    data = base_rms_data
                else:
                    print(f"Reading in {file_path}...")
                    if ext == ".fits":
                        data = hp.read_map(file_path, nest=False)
                    else:
                        data = np.loadtxt(file_path)
                self._map_collections[map_type] = data

            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"File not found for map type '{map_type}'."
                    f" Expected at: {file_path}"
                ) from e

            except Exception as e:
                raise RuntimeError(
                    f"Cannot read file for map type '{map_type}' at {file_path}."
                    f" Underlying error: {e}"
                ) from e
        return

    @property
    def map_collections(self) -> dict[str, Any] | list[dict[str, Any]]:
        '''
        Load and return the specified SKA map collections based on the
        configured settings.
        
        :return: Dictionary of loaded maps.
        '''
        if self._map_collections:
            return self._map_collections
        self.load()
        return self._map_collections

    def clear_cache(self, include_discovery: bool = False) -> None:
        '''
        Clear loaded map collections from memory. Optionally also clear the
        discovery cache so subsequent loads rescan the filesystem.
        '''
        self._map_collections = {}
        if include_discovery:
            self._available_cache = None
