import pickle


class Classifier:
    """This class implements an classifier.

    Attributes:
        model (mlflow.pyfunc.PyFuncModel): Model loaded from mlflow.
    """

    def __init__(self, model):
        if isinstance(model, str):
            self.load(model)
        else:
            self.model = model

    def load(self, input_path):
        """Load model from input path.

        Args:
            input_path (str): Path to serialized .pkl file.
        """
        with open(input_path, "rb") as file:
            self.model = pickle.load(file)

    def predict(self, input):
        """Predict label for image.

        Args:
            input (list[np.array]): features to predict

        Returns:
            The label predicted by the model.
        """
        if not input:
            return None
        pred_label = self.model["clf"].predict([input])
        return str(pred_label[0])
