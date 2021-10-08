from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch.nn.functional as F


class Evaluation:
    def __init__(self, model, eval_iter, device):
        self.model = model
        self.eval_iter = eval_iter
        self.device = device

    def eval_classification(self):
        self.model.eval()
        corrects, total_loss = 0, 0

        for batch in self.eval_iter:
            x, y = batch.text.to(self.device), batch.label.to(self.device)
            logit = self.model(x)
            loss = F.cross_entropy(logit, y, reduction='sum')
            total_loss += loss.item()
            corrects += (logit.max(1)[1].view(y.size()).data == y.data).sum()
            
        size = len(self.eval_iter.dataset)
        avg_loss = total_loss / size
        avg_accuracy = 100.0 * corrects / size
        return avg_loss, avg_accuracy