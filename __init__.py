if __package__:
    from .ponyxl import PonyXL
    from .flux import Flux
else:
    from ponyxl import PonyXL
    from flux import Flux

NODE_CLASS_MAPPINGS = {
    "PonyXL": PonyXL,
    "Flux": Flux
}
print("\033[34mGrok Prompts: \033[92mLoaded\033[0m")
