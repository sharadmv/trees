
def tree_cache(name):
    def func(tree_func):
        def foo(self, *args, **kwargs):
            kw = tuple(kwargs.items())
            key = (name, args, kw)
            if key not in self.cache:
                self.cache[key] = tree_func(self)
            return self.cache[key]
        return foo
    return func
