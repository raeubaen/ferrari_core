ROUTINE_REGISTRY = {}

def get_routine(name):
    return ROUTINE_REGISTRY[name]

def register_routine(name):
    def wrapper(fn):
        ROUTINE_REGISTRY[name] = fn
        return fn
    return wrapper
