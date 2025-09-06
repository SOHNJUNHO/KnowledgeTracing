import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

def preprocess():
    pass

class KTDataset(Dataset):
    def __init__(self, skills, problems, skill_interactions, problem_interactions, answers, seq_len):
        super(KTDataset, self).__init__()
        self.skills = skills
        self.problems = problems
        self.skill_interactions = skill_interactions
        self.problem_interactions = problem_interactions
        self.answers = answers
        self.seq_len = seq_len

        # Split sequences into chunks of seq_len, cut if it is too long
        self.data = [
            (s[i:i+seq_len], p[i:i+seq_len], si[i:i+seq_len], pi[i:i+seq_len], a[i:i+seq_len])
            for s, p, si, pi, a in zip(skills, problems, skill_interactions, problem_interactions, answers)
            for i in range(0, len(s), seq_len)
        ]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def load_data(df, batch_size, seq_len):

    skill_ids = [torch.tensor(u_df["skill_id"].values, dtype=torch.long)
                for _, u_df in df.groupby("user_id")]
    problem_ids = [torch.tensor(u_df["problem_id"].values, dtype=torch.long)
                for _, u_df in df.groupby("user_id")]
    skill_inter_ids = [torch.tensor(u_df["skill_with_answer"].values, dtype=torch.long)
                 for _, u_df in df.groupby("user_id")]            
    pro_inter_ids = [torch.tensor(u_df["problem_with_answer"].values, dtype=torch.long)
                 for _, u_df in df.groupby("user_id")]
    answer = [torch.tensor(u_df["correct"].values, dtype=torch.long)
              for _, u_df in df.groupby("user_id")]

    # One step behind...excluding the last element
    skill_inter_ids = [torch.cat((torch.zeros(1, dtype=torch.long), s))[:-1] for s in skill_inter_ids]
    pro_inter_ids = [torch.cat((torch.zeros(1, dtype=torch.long), s))[:-1] for s in pro_inter_ids]

    kt_dataset = KTDataset(skill_ids, problem_ids, skill_inter_ids, pro_inter_ids, answer, seq_len)
    
    def pad_collate(batch):
        (skill_ids, problem_ids, skill_inter_ids, pro_inter_ids, answer) = zip(*batch)
        skill_ids = pad_sequence(skill_ids, batch_first=True, padding_value= 0)
        problem_ids = pad_sequence(problem_ids, batch_first=True, padding_value= 0)
        skill_inter_ids = pad_sequence(skill_inter_ids, batch_first=True, padding_value= 0)
        pro_inter_ids = pad_sequence(pro_inter_ids, batch_first=True, padding_value= 0)
        answer = pad_sequence(answer, batch_first=True, padding_value=-1)
        return skill_ids, problem_ids, skill_inter_ids, pro_inter_ids, answer

   # Get the total size of the dataset after splitting sequences
    total_size = len(kt_dataset)
    
    train_ratio = 0.7
    val_ratio = 0.2
    
    # Calculate the sizes for train, validation, and test splits
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - (train_size + val_size)  # Ensure the sizes sum up to total_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(kt_dataset, [train_size, val_size, test_size])
    
    train_data_loader = DataLoader(train_dataset, batch_size= batch_size, shuffle = True, collate_fn=pad_collate)
    valid_data_loader = DataLoader(val_dataset, batch_size= batch_size, shuffle = False, collate_fn=pad_collate)
    test_data_loader = DataLoader(test_dataset, batch_size= batch_size, shuffle = False, collate_fn=pad_collate)

    return train_data_loader, valid_data_loader, test_data_loader

