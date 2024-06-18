# __contains__ works with the in keyword

class MyContainer:
    def __init__(self, bounds: tuple):
        assert len(bounds) == 2
        self._bounds = bounds

    def __contains__(self, item):
        return self._bounds[0] <= item <= self._bounds[1]


my_container = MyContainer((0, 180))
assert 90 in my_container


# __getattr__ provides a way to call dynamic attributes
class DynamicAttributes:

    def __init__(self, attribute):
        self.attribute = attribute

    def __getattr__(self, attr):
        if attr.startswith("fallback_"):
            name = attr.replace("fallback_", "")
            return f"[fallback resolved] {name}"
        raise AttributeError(
            f"{self.__class__.__name} has no attribute {attr}"
        )

dyn = DynamicAttributes("value")
print(dyn.attribute)
print(dyn.fallback_test)

dyn.__dict__["fallback_new"] = "new value"
print(dyn.fallback_new)

# This needs to raise an attribute error to work
print(getattr(dyn, "something", "default"))

# __call__ makes the object a callable like a func
from collections import defaultdict

class CallCount:
    def __init__(self):
        self._counts = defaultdict(int)

    def __call__(self, argument):
        self._counts[argument] += 1
        return self._counts[argument]

cc = CallCount()

print(cc(1))
print(cc(2))
print(cc(1))
print(cc(1))
print(cc("something"))
print(callable(cc))