class ListenerInterface:

    def on_event(self, values: dict):
        """
        The method that is called when an event is triggered.

        :param values: The values that are updated.
        """
        raise NotImplementedError