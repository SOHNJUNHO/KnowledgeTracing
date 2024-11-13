import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


def preprocess():
    pass

#df= pd.read_csv('data/skill_builder_data.csv')

#df = df[['user_id','problem_id','skill_id','skill_name','correct']]

#df.dropna(subset=['skill_id', 'problem_id'], inplace=True)

#df = df.groupby('user_id').filter(lambda q: len(q) > 1).copy()

#df['skill'], _ = pd.factorize(df['skill_id'], sort=True)  # we can also use problem_id to represent exercises
#df['problem'],_ = pd.factorize(df['problem_id'], sort=True)

#df['skill'] = df['skill'].apply(lambda x:x+1) 
#df['problem'] = df['problem'].apply(lambda x:x+1) 

#df['skill_with_answer'] = df['skill'] +  len(df['skill'].unique()) * df['correct']  #q + self.num_q * r
#df['problem_with_answer'] = df['problem'] + len(df['problem'].unique()) * df['correct']



class KTDataset(Dataset):
    def __init__(self, features, questions, answers, seq_len):
        super(KTDataset, self).__init__()
        self.features = features
        self.questions = questions
        self.answers = answers
        self.seq_len = seq_len

        # Flatten and split long sequences
        self.data = []
        for feat, qst, ans in zip(features, questions, answers):
            for i in range(0, len(feat), self.seq_len):
                self.data.append((
                    feat[i:i+self.seq_len],
                    qst[i:i+self.seq_len],
                    ans[i:i+self.seq_len]
                ))

    def __getitem__(self, index):
        return self.data[index]


    def __len__(self):
        return len(self.data)


def load_data(df, batch_size, seq_len):

    problem_ids = [torch.tensor(u_df["problem"].values, dtype=torch.long)
                for _, u_df in df.groupby("user_id")]
    interaction_ids = [torch.tensor(u_df["problem_with_answer"].values, dtype=torch.long)
                 for _, u_df in df.groupby("user_id")]
    answer = [torch.tensor(u_df["correct"].values, dtype=torch.long)
              for _, u_df in df.groupby("user_id")]

    # One step behind...
    interaction_ids = [torch.cat((torch.zeros(1, dtype=torch.long), s))[:-1] for s in interaction_ids]


    kt_dataset = KTDataset(problem_ids, interaction_ids, answer, seq_len)
    
    def pad_collate(batch):
        (problem_ids, interaction_ids, answer) = zip(*batch)
        problem_ids = pad_sequence(problem_ids, batch_first=True, padding_value= 0)
        interaction_ids = pad_sequence(interaction_ids, batch_first=True, padding_value=0)
        answer = pad_sequence(answer, batch_first=True, padding_value=-1)
        return problem_ids, interaction_ids, answer

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

