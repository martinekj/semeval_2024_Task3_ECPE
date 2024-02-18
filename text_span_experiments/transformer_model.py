import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertModel
import numpy as np

import os


class BertTextSpansModel(nn.Module):

    def __init__(self, modelid, device=None, out_dim=None):
        super().__init__()
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.out_dim = out_dim
        self.modelid = modelid
        self.model = BertModel.from_pretrained(modelid)
        self.model.to(self.device)
        # classifier head
        self.final_head = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, out_dim)
        )
        self.final_head.to(self.device)

    def forward(self, input_ids, attention_mask):

        x = self.model(input_ids.to(self.device), attention_mask.to(self.device)) # forward with prepared embedding and mask

        # cls token
        cls = x.last_hidden_state[:, 0]
        return self.final_head(cls)


    def predict_classes(self, dataloader):
        print(f"Predicting {len(dataloader)} samples")
        predicted_classes = []
        self.train(False)
        with torch.no_grad():
            counter = 0
            for x, xm, y in dataloader:
                # y = y.to(self.device)
                pred = self(x, xm)  # [3,26]
                predicted_index = pred.argmax(1).detach().cpu().numpy()[0]
                predicted_classes.append(predicted_index)
                if counter % 100 == 0:
                    print(f"Counter: {counter}/{len(dataloader)}")
                counter += 1
        return predicted_classes

    def loop_test(self, dataloader):
        self.train(False)

        total_steps = len(dataloader)
        total_loss = 0.0
        correct, total = 0, 0
        tp, fp, fn = 0, 0, 0
        true_positives, false_positives, false_negatives = np.zeros(shape=self.out_dim), np.zeros(shape=self.out_dim), np.zeros(shape=self.out_dim)
        print("Testing model...")
        with torch.no_grad():
            counter = 0
            for x, xm, y in dataloader:
                y = y.to(self.device)

                pred = self(x, xm)  # [3,26]

                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                total += torch.numel(y)

                lossv = self.criterion(pred, y).item()
                total_loss += lossv


                counter += 1
                if counter % 10 == 0:
                    print("batch processed: ", counter, "/", len(dataloader))



        mloss = total_loss / total_steps
        acc = correct / total

        print(f"Val acc: {(100 * acc):>0.1f}%, avg loss: {mloss:>8f} \n")
        return {
            "acc": acc,
            "loss": mloss
        }


    def evaluate_f1(self, test_data):

        dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

        self.train(False)

        total_steps = len(dataloader)
        total_loss = 0.0
        correct, total = 0, 0
        tp, fp, fn = 0, 0, 0
        epsilon = 0.00000000000001
        true_positives, false_positives, false_negatives = np.zeros(shape=self.out_dim), np.zeros(shape=self.out_dim), np.zeros(
            shape=self.out_dim)
        true_positives += epsilon
        false_positives += epsilon
        false_negatives += epsilon
        print("Evaluating model, calculating macro f1...")
        with torch.no_grad():
            counter = 0
            for x, xm, y in dataloader:
                y = y.to(self.device)

                pred = self(x, xm)  # [3,26]

                predicted_label = torch.argmax(pred.flatten()).item()
                ground_truth_label = y.item()

                if predicted_label == ground_truth_label:
                    true_positives[ground_truth_label] += 1
                else:
                    false_positives[predicted_label] += 1
                    false_negatives[ground_truth_label] += 1

                counter += 1
                if counter % 100 == 0:
                    print("Samples processed: ", counter, "/", len(dataloader))


        micro_precision = np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_positives))
        micro_recall = np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_negatives))
        micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall)

        macro_precision = np.average(np.divide(true_positives, np.add(true_positives, false_positives)))
        macro_recall = np.average(np.divide(true_positives, np.add(true_positives, false_negatives)))
        macro_f1 = (2 * macro_precision * macro_recall) / (macro_precision + macro_recall)

        print("Macro F1 = ", macro_f1)
        return macro_f1,

    def loop_train(self, dataloader):
        self.train(True)
        size = len(dataloader.dataset)
        total_loss, total_steps = 0.0, 0
        correct, total = 0, 0

        for batch, (x, xm, y) in enumerate(dataloader):
            y = y.to(self.device)

            # Compute prediction and loss
            pred = self(x, xm)
            loss = self.criterion(pred, y)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # predt = pred >= 0  # no sigmoid
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            total += torch.numel(y)

            lossv = loss.item()
            total_loss += lossv
            total_steps += 1

            if batch % 10 == 0:
                current = batch * len(x)
                print(
                    f"[{current:>5d}/{size:>5d}] \t avg loss: {total_loss / total_steps:>7f} \t acc: {100 * (correct / total):>0.1f}%")

        mloss = total_loss / total_steps
        acc = correct / total

        print(f"Train acc: {(100 * acc):>0.1f}%, avg loss: {mloss:>8f} \n")
        return {
            "acc": acc,
            "loss": mloss
        }


    def train_model(self, train_dataset, val_dataset,  epochs, criterion, optimizer, batch):
        print("Training model for", epochs, "epochs")

        self.optimizer = optimizer
        self.criterion = criterion

        # Prepare data
        train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch, shuffle=False)

        modeldir = "text_spans_models/"+self.modelid+"/"
        if not os.path.exists(modeldir):
            os.makedirs(modeldir)

        bestmodelfile = f"{modeldir}/model_best_val_acc.cp"

        best_res = 0.
        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}\n-------------------------------")
            train_res = self.loop_train(train_dataloader)
            val_res = self.loop_test(val_dataloader)
            if val_res["acc"] > best_res:
                best_res = val_res["acc"]
                self.save_learnable_params(bestmodelfile)

        print('Training Finished')
        # save model from the last epoch
        self.save_learnable_params(f'{modeldir}/model_{epochs}.cp')

        return train_res, val_res


    def test_model(self, test_X, test_att_mask_X, test_y):
        print("Testing model")
        self.eval()
        # X is a torch Variable

        correct = 0

        permutation = torch.randperm(test_X.size()[0])
        testing_steps = 0
        total_steps = len(test_X)
        total = 0
        # print("Total steps:", total_steps)

        batch_size = 1
        for i in range(0, test_X.size()[0], batch_size):
            indices = permutation[i:i + batch_size]
            test_sample_x, test_sample_att_mask_x, test_sample_y = test_X[indices], test_att_mask_X[indices], test_y[indices]

            outputs = self.forward(test_sample_x, test_sample_att_mask_x)
            pred = torch.max(outputs, dim=1)[1]

            correct += (pred == test_sample_y).float()
            total += 1
            testing_steps += 1

        acc = (correct / total).item()

        print("______________________________________________________________")
        print("FINAL EVALUATION:")
        print("\tAccuracy: ", acc)
        print("______________________________________________________________")

        print('Test Passed')
        return acc

    def save_learnable_params(self, path):
        sd = self.state_dict()
        rmkeys = [name for name, param in self.named_parameters() if not param.requires_grad]
        for k in rmkeys:
            sd.pop(k)
        torch.save(sd, path)

    def load_learnable_params(self, path):
        mk = self.load_state_dict(torch.load(path, map_location=self.device), strict=False)
        for k in mk.missing_keys:
            assert k.startswith("model.")

    def get_num_of_learnable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
