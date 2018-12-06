class Namespace:
    @classmethod
    def get_all_names(cls):
        return list(cls._name_to_value.keys())

    @classmethod
    def get_value(cls, name):
        return cls._name_to_value[name]
