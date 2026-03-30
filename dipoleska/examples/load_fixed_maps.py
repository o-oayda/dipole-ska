from dipoleska.utils.map_read import MapCollectionLoader


loader = MapCollectionLoader(
    base_dirs=['data/ska/fixed_maps_29-01-26/THISONE_FluxFinal_ModifiedSizes']
)
map_dicts = loader.map_collections

print(f'Loaded {len(map_dicts)} map collections.')
