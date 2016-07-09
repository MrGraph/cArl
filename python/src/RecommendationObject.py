from ArtificialNeuralNetwork import ArtificialNeuralNetwork



class RecommendationObject(Object):
    
    def __init__(self):
        self.ann = ArtificialNeuralNetwork()
        self.provenance_dict = {}
