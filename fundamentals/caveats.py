# Mutable default args are bad
def wrong_user_display(user_metadata: dict = {"name": "John", "age": 30}):
    name = user_metadata.pop("name")
    age = user_metadata.pop("age")

    return f"{name} ({age})"

# print(wrong_user_display())
# print(wrong_user_display({"name": "Jane", "age": 25}))
# print(wrong_user_display())

# correct way: use None as a sentinel default, set it up in function body instead

# Do not extend lists, strings, dicts etc. directly. Use collections as a parent class
