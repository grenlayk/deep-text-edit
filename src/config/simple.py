class Config:
    def __init__(self):
        self.model = ...
        self.criterion = ...
        self.optimizer = ...
        self.dataset = Dataset(local)
        self.trainer = SimpleTrainer()

    def run():
        self.trainer.run()
