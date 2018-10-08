class Rules:
    @staticmethod
    def actions(state) -> []:
        raise NotImplementedError()

    @staticmethod
    def next(state, action):
        raise NotImplementedError()

    @staticmethod
    def undo(state, last_action):
        raise NotImplementedError()
