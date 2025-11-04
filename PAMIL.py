# Adapted from https://github.com/Jiashuai-Liu/PAMIL

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import argparse
import random
from sklearn.metrics import f1_score

parser = argparse.ArgumentParser(description="PAMIL - Prototype-based Multiple Instance Learning")
parser.add_argument("--emb_train", type=str, required=True, help="Path to train embeddigs directory")
parser.add_argument("--emb_test", type=str, required=True, help="Path to test embeddigs directory")
parser.add_argument("--nproto", type=int, required=True, help="Number of prototypes to be used")
parser.add_argument("--dim", type=int, required=True, help="Encoder output dimension")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--size_arg", type=str, default='small', choices=['small', 'big'], 
                    help="Model size")
parser.add_argument("--dropout", action='store_true', help="Use dropout")
parser.add_argument("--proto_pred", action='store_true', help="Use prototype prediction")
parser.add_argument("--proto_weight", type=float, default=0.5, help="Weight for prototype prediction")
parser.add_argument("--inst_pred", action='store_true', help="Use instance prediction")
parser.add_argument("--k_sample", type=int, default=8, help="Number of samples for instance eval")
parser.add_argument("--lambda_clst", type=float, default=0.005, help="Weight for clustering loss")
parser.add_argument("--lambda_proto_clst", type=float, default=0.005, help="Weight for prototype clustering loss")
parser.add_argument("--lambda_er", type=float, default=0.1, help="Weight for ER loss")
parser.add_argument("--lambda_inst", type=float, default=0.5, help="Weight for instance loss")
parser.add_argument("--seed",type=int, default=1)
parser.add_argument("--resume_path", type=str, default=None, help="Path to pretrained model for curriculum learning (e.g., WR=10%)")
parser.add_argument("--freeze_epochs", type=int, default=5, help="Number of epochs to freeze prototypes and fc before fine-tuning")

args = parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(seed=args.seed)


class PreEmbeddedBagDataset(Dataset):
    def __init__(self, embeddings_dir, label_dict, transform=None):
        """
        embeddings_dir: path to folder containing .npy files
        label_dict: dict {bag_name: 0/1}
        """
        # Get all .npy files and extract bag names
        self.embedding_files = [f for f in os.listdir(embeddings_dir) if f.endswith('_features.npy')]
        self.bag_names = [f.replace('_features.npy', '') for f in self.embedding_files]
        self.embeddings_dir = embeddings_dir
        self.labels = [label_dict[bag_name] for bag_name in self.bag_names]
        self.transform = transform

    def __len__(self):
        return len(self.bag_names)

    def __getitem__(self, idx):
        bag_name = self.bag_names[idx]
        embedding_file = self.embedding_files[idx]
        embedding_path = os.path.join(self.embeddings_dir, embedding_file)
        
        # Load pre-extracted features
        data = np.load(embedding_path, allow_pickle=True).item()
        features = data['feature']  
        
        if len(features.shape) == 1:
            feature_dim = args.dim  
            num_patches = features.shape[0] // feature_dim
            features = features.reshape(num_patches, feature_dim)
        
        features = torch.from_numpy(features).float()
        
        label = torch.tensor(self.labels[idx]).long()
        return features, label, bag_name


# Gated Attention Network
class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_proto=10, n_classes=2):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [nn.Linear(L, D), nn.Tanh()]
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(n_proto, n_classes)

    def forward(self, proto, x):
        a = self.attention_a(proto)  # p * D
        b = self.attention_b(x)      # n * D
        proto_A = torch.mm(a, b.T)   # p * n
        proto_A = torch.transpose(proto_A, 0, 1)  # n * p
        A = self.attention_c(proto_A)  # n * n_classes
        return proto_A, A
    

class ERLoss(nn.Module):
    def __init__(self):
        super(ERLoss, self).__init__()
        self.kl = nn.KLDivLoss(reduction='batchmean')

    def forward(self, bag_logits, proto_logits):
        bag_prob = F.softmax(bag_logits, dim=1)
        proto_prob = F.softmax(proto_logits, dim=1)
        loss = self.kl(torch.log(bag_prob + 1e-8), proto_prob.detach())
        return loss

# PAMIL Model
class PAMIL(nn.Module):        
    def __init__(self, gate=True, size_arg='small', dropout=False, n_protos=10, 
                 n_classes=2, proto_path=None, proto_pred=False, proto_weight=0.5,
                 inst_pred=False, k_sample=8):
        super(PAMIL, self).__init__()
        self.size_dict = {"small": [args.dim, 512, 256], "big": [args.dim, 512, 384]}       
        size = self.size_dict[size_arg]
        self.n_protos = n_protos
        self.n_classes = n_classes
        self.proto_pred = proto_pred
        self.inst_pred = inst_pred
        self.k_sample = k_sample
        
        # Initialize prototypes
        if proto_path and os.path.exists(proto_path):
            proto_init = torch.from_numpy(np.load(proto_path))
            self.proto = nn.Parameter(proto_init, requires_grad=True)
        else:
            print("Randomly initializing prototype vectors")
            self.proto = nn.Parameter(torch.randn(n_protos, size[0]), requires_grad=True)
        
        # Feature projection
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        self.fc = nn.Sequential(*fc)
            
        self.attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, 
                                            n_proto=n_protos, n_classes=n_classes)
        
        # Bag-level classifiers (one per class)
        #bag_classifiers = [nn.Linear(size[1], 1) for i in range(n_classes)]
        bag_classifiers = [nn.utils.parametrizations.weight_norm(nn.Linear(size[1], 1)) for _ in range(n_classes)]
        self.classifiers = nn.ModuleList(bag_classifiers)
        
        # Prototype prediction
        if proto_pred:
            self.proto_weight = proto_weight
            proto_pred_classifier = [nn.Linear(size[1], n_classes)]
            self.proto_pred_classifier = nn.Sequential(*proto_pred_classifier)
            self.ERloss_function = ERLoss()
        
        # Instance prediction
        if inst_pred:
            instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
            self.instance_classifiers = nn.ModuleList(instance_classifiers)
            self.instance_loss_fn = nn.CrossEntropyLoss()
        
    def inst_eval(self, A, h, classifier): 
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        k = min(self.k_sample, A.shape[-1])
        
        top_p_ids = torch.topk(A, k)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, k, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        
        p_targets = self.create_positive_targets(k, device)
        n_targets = self.create_negative_targets(k, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets
    
    def inst_eval_out(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        k = min(self.k_sample, A.shape[-1])
        
        top_p_ids = torch.topk(A, k)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(k, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets
    
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device).long()
    
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device).long()
        
    def forward(self, h, label=None, return_features=False):
        result_dict = {}
        device = h.device
        
        # Project prototypes and features
        proto = F.normalize(self.fc(self.proto), p=2, dim=1)
        h = F.normalize(self.fc(h), p=2, dim=1)
        
        # Gated attention
        proto_A, A_raw = self.attention_net(proto, h)  # proto_A: n * p, A_raw: n * c
        A_raw = torch.transpose(A_raw, 1, 0)  # c * n
        A = F.softmax(A_raw, dim=1)  # softmax over n
        M = torch.mm(A, h)  # c * d
        M = F.normalize(M, p=2, dim=1)
        
        # Bag-level predictions (one classifier per class)
        logits = torch.empty(1, self.n_classes).float().to(device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])
        
        # Prototype prediction
        if self.proto_pred:
            proto_score, _ = torch.max(proto_A, dim=0)  # 1 * p
            proto_score = proto_score / torch.max(proto_score)
            proto_score = F.softmax(proto_score.unsqueeze(0), dim=1)
            proto_M = torch.mm(proto_score, proto)
            proto_logits = self.proto_pred_classifier(proto_M)
            
            loss_er = self.ERloss_function(logits, proto_logits)
            logits = logits * (1 - self.proto_weight) + proto_logits * self.proto_weight
            result_dict['loss_er'] = loss_er
        
        # Instance prediction
        if self.inst_pred and label is not None:
            loss_inst = torch.tensor(0.).to(device)
            for c in range(self.n_classes):
                inst_label = label[0, c] if len(label.shape) > 1 else (label.item() == c)
                instance_classifier = self.instance_classifiers[c]
                if inst_label == 1:
                    instance_loss, preds, targets = self.inst_eval(A, h, instance_classifier)
                else:
                    instance_loss, preds, targets = self.inst_eval_out(A, h, instance_classifier)
                loss_inst += instance_loss
            loss_inst /= self.n_classes
            result_dict['loss_inst'] = loss_inst
        
        # Clustering losses
        # Feature-to-prototype clustering (minimize max attention per instance)
        loss_clst = 1 - torch.mean(torch.max(F.softmax(proto_A, dim=1), dim=1)[0])
        # Prototype-to-feature clustering (minimize max attention per prototype)
        loss_proto_clst = 1 - torch.mean(torch.max(F.softmax(proto_A, dim=1), dim=0)[0])

        
        # Predictions
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.argmax(logits, dim=1)
 
        result_dict['A_raw'] = A_raw
        result_dict['proto_A'] = proto_A
        result_dict['features'] = M
        result_dict['loss_clst'] = loss_clst
        result_dict['loss_proto_clst'] = loss_proto_clst
        
        if return_features:
            return logits, Y_prob, Y_hat, result_dict, h
        
        return logits, Y_prob, Y_hat, result_dict

def train_model(embeddings_dir, label_dict, num_classes=2, num_prototypes=10, 
                epochs=20, lr=1e-3, input_dim=args.dim, save_path='pamil_preembedded.pth'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    dataset = PreEmbeddedBagDataset(embeddings_dir, label_dict)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Initialize model with reference architecture
    model = PAMIL(
        size_arg=args.size_arg,
        dropout=args.dropout,
        n_protos=args.nproto,
        n_classes=num_classes,
        proto_pred=args.proto_pred,
        proto_weight=args.proto_weight,
        inst_pred=args.inst_pred,
        k_sample=args.k_sample
    ).to(device)

    if args.resume_path is not None and os.path.exists(args.resume_path):
        print(f"\n[Curriculum] Loading pretrained model from {args.resume_path}")
        state_dict = torch.load(args.resume_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print("[Curriculum] Weights loaded successfully!")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"Training on {len(dataset)} bags")
    print(f"Device: {device}")
    print(f"Proto pred: {args.proto_pred}, Inst pred: {args.inst_pred}")
    
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_loss_ce = 0
        total_loss_clst = 0
        total_loss_proto_clst = 0
        total_loss_er = 0
        total_loss_inst = 0
        correct = 0
        total = 0

        # Freeze prototypes and fc layers for the first few epochs if curriculum learning
        if args.resume_path is not None and epoch < args.freeze_epochs:
            for name, param in model.named_parameters():
                if "proto" in name or "fc" in name:
                    param.requires_grad = False
            if epoch == 0:
                print(f"[Curriculum] Freezing proto & fc for first {args.freeze_epochs} epochs")
        else:
            for param in model.parameters():
                param.requires_grad = True

        
        for features, label, bag_name in loader:
            features = features.squeeze(0).to(device)  # (num_patches, D)
            label = label.to(device)
            
            # Convert to multi-label format if using inst_pred
            if args.inst_pred:
                label_onehot = torch.zeros(1, num_classes).to(device)
                label_onehot[0, label.item()] = 1
            else:
                label_onehot = None
            
            # Forward pass
            logits, Y_prob, Y_hat, result_dict = model(features, label=label_onehot)
            
            # Main classification loss
            loss_ce = criterion(logits, label)

            # Clustering losses
            loss_clst = result_dict['loss_clst']
            loss_proto_clst = result_dict['loss_proto_clst']
            
            # Total loss
            loss = loss_ce + args.lambda_clst * loss_clst + args.lambda_proto_clst * loss_proto_clst
            
            # Optional losses
            if args.proto_pred and 'loss_er' in result_dict:
                loss_er = result_dict['loss_er']
                loss += args.lambda_er * loss_er
                total_loss_er += loss_er.item()
            
            if args.inst_pred and 'loss_inst' in result_dict:
                loss_inst = result_dict['loss_inst']
                loss += args.lambda_inst * loss_inst
                total_loss_inst += loss_inst.item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            total_loss_ce += loss_ce.item()
            total_loss_clst += loss_clst.item()
            total_loss_proto_clst += loss_proto_clst.item()
            
            # Calculate accuracy
            total += 1
            correct += (Y_hat == label).sum().item()
            
        # Print epoch statistics
        avg_loss = total_loss / len(loader)
        avg_loss_ce = total_loss_ce / len(loader)
        avg_loss_clst = total_loss_clst / len(loader)
        avg_loss_proto_clst = total_loss_proto_clst / len(loader)
        accuracy = 100 * correct / total
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Total Loss: {avg_loss:.4f}")
        print(f"  CE Loss: {avg_loss_ce:.4f}")
        print(f"  Clst Loss: {avg_loss_clst:.4f}")
        print(f"  Proto Clst Loss: {avg_loss_proto_clst:.4f}")
        
        if args.proto_pred:
            print(f"  ER Loss: {total_loss_er / len(loader):.4f}")
        if args.inst_pred:
            print(f"  Inst Loss: {total_loss_inst / len(loader):.4f}")
        
        print(f"  Accuracy: {accuracy:.2f}%")

        if accuracy >= best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
    if args.resume_path:
        print(f"\n[Curriculum] Fine-tuning complete. New model saved to {save_path}\n")

"""        if accuracy >= 100.0:
            break"""


def run_inference_all(model_path, bags_root, output_csv):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load trained model
    model = PAMIL(
        size_arg=args.size_arg,
        dropout=args.dropout,
        n_protos=args.nproto,
        n_classes=2,
        proto_pred=args.proto_pred,
        proto_weight=args.proto_weight,
        inst_pred=args.inst_pred,
        k_sample=args.k_sample
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    all_records = []

    for fname in sorted(os.listdir(bags_root)):
        if not fname.endswith("_features.npy"):
            continue
        bag_name = fname.replace("_features.npy", "")
        bag_path = os.path.join(bags_root, fname)

        # Load bag features
        data = np.load(bag_path, allow_pickle=True).item()
        features = data["feature"]  
        file_name = data["file_name"]

        print(f"Loaded {bag_name}: raw features shape = {features.shape}")
        
        # If features are 1D, we need to reshape them
        if len(features.shape) == 1:
            feature_dim = args.dim 
            num_patches = features.shape[0] // feature_dim
            features = features.reshape(num_patches, feature_dim)
            print(f"Reshaped to: {features.shape}")
        
        features = torch.from_numpy(features).float().to(device) 

        with torch.no_grad():
            logits, Y_prob, Y_hat, result_dict = model(features)
            pred_label = Y_hat.item()
            pred_prob = Y_prob.cpu().numpy()

        # Get attention weights
        A_raw = result_dict['A_raw']
        attn_weights = F.softmax(A_raw, dim=1)
        attn_weights = attn_weights.t().cpu().numpy()  # n * c

        # Collect results for this bag
        for patch_id in range(len(file_name)):
            record = {
                "bag_id": bag_name,
                "patch_id": file_name[patch_id],
                "bag_pred": pred_label,
            }
            
            # Add per-class attention and probabilities
            for c in range(attn_weights.shape[1]):
                record[f"attn_class_{c}"] = float(attn_weights[patch_id, c])
                record[f"bag_prob_class_{c}"] = float(pred_prob[0, c])
            
            # For binary classification
            if attn_weights.shape[1] == 2:
                record["attn_normal"] = float(attn_weights[patch_id, 0])
                record["attn_abnormal"] = float(attn_weights[patch_id, 1])
                record["bag_prob_normal"] = float(pred_prob[0, 0])
                record["bag_prob_abnormal"] = float(pred_prob[0, 1])
            
            all_records.append(record)

        print(f"Processed {bag_name}: predicted label {pred_label} (prob: {pred_prob[0][pred_label]:.3f})")

    df = pd.DataFrame(all_records)
    df.to_csv(output_csv, index=False)
    print(f"Saved all bag attention scores -> {output_csv}")

    print("\nSummary:")
    print(f"Total bags processed: {len(df['bag_id'].unique())}")
    print(f"Total patches: {len(df)}")
    pred_counts = df.groupby('bag_id')['bag_pred'].first().value_counts()
    print(f"Prediction distribution: {dict(pred_counts)}")

def analyze_results(csv_path):
    # Load data
    df = pd.read_csv(csv_path)

    # Extract true label (abnormal=1, normal=0) from bag_id
    df["true_label"] = df["bag_id"].apply(lambda x: 1 if x.startswith("abnormal") else 0)

    # Compute F1-score (bag-level, across all bags)
    bag_preds = df.groupby("bag_id").first()[["true_label", "bag_pred"]]
    f1 = f1_score(bag_preds["true_label"], bag_preds["bag_pred"])

    print(f"F1-score (predicted vs true for ALL bags): {f1:.4f}")

    # Misclassified bags
    wrong_bags = bag_preds[bag_preds["true_label"] != bag_preds["bag_pred"]].index.tolist()
    if wrong_bags:
        print("\nMisclassified bags:")
        for b in wrong_bags:
            print(f"- {b}")
    else:
        print("\nNo misclassified bags")

    # Non-LYT count in Top-400 for each abnormal bag
    if 'attn_abnormal' in df.columns:
        print("\nNon-LYT in Top-400 (per abnormal bag):")
        for bag_id, bag_df in df[df["true_label"] == 1].groupby("bag_id"):
            top_patches = bag_df.sort_values("attn_abnormal", ascending=False).head(400)
            non_lyt_count = (~top_patches["patch_id"].str.startswith("LYT")).sum()
            print(f"{bag_id}: {non_lyt_count} / 400")


def create_example_labels(embeddings_dir):
    """
    Create example labels based on bag names
    Modify this function according to your actual labeling scheme
    """
    embedding_files = [f for f in os.listdir(embeddings_dir) if f.endswith('_features.npy')]
    bag_names = [f.replace('_features.npy', '') for f in embedding_files]
    
    # Example labeling scheme - modify as needed
    label_dict = {}
    for bag_name in bag_names:
        # Example: assign labels based on some pattern in bag name
        # You should replace this with your actual labeling logic
        if 'abnormal' in bag_name.lower() or 'malignant' in bag_name.lower():
            label_dict[bag_name] = 1
        else:
            label_dict[bag_name] = 0
    
    return label_dict


if __name__ == '__main__':
        
        label_dict = create_example_labels(args.emb_train)
        
        print(f"Found {len(label_dict)} bags with labels")
        print("Label distribution:", {k: sum(1 for v in label_dict.values() if v == k) for k in set(label_dict.values())})
        
        # Train the model
        train_model(
            embeddings_dir=args.emb_train,
            label_dict=label_dict,
            input_dim=args.dim,
            num_classes=2,
            num_prototypes=args.nproto,
            epochs=args.epochs,
            lr=args.lr,
            save_path=f'pamil_model.pth'
        )
        
        run_inference_all(
            model_path=f"pamil_model.pth",
            bags_root=args.emb_test,
            output_csv=f"./results_test.csv"
        )
        
        csv_path = f"./results_test.csv"
        analyze_results(csv_path)