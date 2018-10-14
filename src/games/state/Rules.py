class Rules:

    @staticmethod
    def initial_state():
        raise NotImplementedError()

    @staticmethod
    def actions(state) -> []:
        raise NotImplementedError()

    @staticmethod
    def apply(state, action):
        raise NotImplementedError()
