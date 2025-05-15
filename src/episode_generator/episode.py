class Episode:
    def __init__(self, query_tensor, response_tensor, reward, response):
        self.query_tensor = query_tensor
        self.response_tensor = response_tensor
        self.reward = reward
        self.response = response
