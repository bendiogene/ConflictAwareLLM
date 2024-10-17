import json

# Function to compute std
def calculate_std(data):
    
    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
    std_dev = variance ** 0.5
    return std_dev

# Define the path to the JSON file
file_path = 'experiments/gpt2-small/experiment_2/results/experiment2_20240607_095632.json'

# Open and read the JSON file
with open(file_path, 'r') as file:
    data = json.load(file)

res = data['results']
cum_A_fta = 0
cum_B_fta = 0
cum_A_ftb = 0
cum_B_ftb = 0 
cum_A_cftb_raw = 0
cum_B_cftb_raw = 0 
cum_A_cftb_norm = 0
cum_B_cftb_norm = 0 
cum_A_cftb_rnd = 0
cum_B_cftb_rnd = 0 
for f in res:
    acc_A_fta = res[f]['fta']['acc_A']
    acc_B_fta = res[f]['fta']['acc_B']

    acc_A_ftb = res[f]['ftb']['avg_accA']
    acc_B_ftb = res[f]['ftb']['avg_accB']
    std_A_ftb = calculate_std(res[f]['ftb']['acc_A'])
    std_B_ftb = calculate_std(res[f]['ftb']['acc_B'])

    acc_A_cftb_raw = [res[f]['cftb'][t]['raw']['avg_accA'] for t in res[f]['cftb']]
    acc_B_cftb_raw = [res[f]['cftb'][t]['raw']['avg_accB'] for t in res[f]['cftb']]
    std_A_cftb_raw = [calculate_std(res[f]['cftb'][t]['raw']['acc_A']) for t in res[f]['cftb']]
    std_B_cftb_raw = [calculate_std(res[f]['cftb'][t]['raw']['acc_B']) for t in res[f]['cftb']]

    acc_A_cftb_norm = [res[f]['cftb'][t]['norm']['avg_accA'] for t in res[f]['cftb']]
    acc_B_cftb_norm = [res[f]['cftb'][t]['norm']['avg_accB'] for t in res[f]['cftb']]
    std_A_cftb_norm = [calculate_std(res[f]['cftb'][t]['norm']['acc_A']) for t in res[f]['cftb']]
    std_B_cftb_norm = [calculate_std(res[f]['cftb'][t]['norm']['acc_B']) for t in res[f]['cftb']]

    acc_A_cftb_rnd = [res[f]['cftb'][t]['rnd']['avg_accA'] for t in res[f]['cftb']]
    acc_B_cftb_rnd = [res[f]['cftb'][t]['rnd']['avg_accB'] for t in res[f]['cftb']]
    std_A_cftb_rnd = [calculate_std(res[f]['cftb'][t]['rnd']['acc_A']) for t in res[f]['cftb']]
    std_B_cftb_rnd = [calculate_std(res[f]['cftb'][t]['rnd']['acc_B']) for t in res[f]['cftb']]

    print(f"FOLD {f}")
    print(f"FTA -- A: {acc_A_fta:.3f} B: {acc_B_fta:.3f}")
    print(f"FTB -- A: {acc_A_ftb:.3f} ({std_A_ftb:.3f}) B: {acc_B_ftb:.3f} ({std_B_ftb:.3f})")
    for i,t in enumerate(res[f]['cftb']):
        print(f"CFTB t:{t} -- RAW A: {acc_A_cftb_raw[i]:.3f} ({std_A_cftb_raw[i]:.3f}) B: {acc_B_cftb_raw[i]:.3f} ({std_B_cftb_raw[i]:.3f}) - NORM A: {acc_A_cftb_norm[i]:.3f} ({std_A_cftb_norm[i]:.3f}) B: {acc_B_cftb_norm[i]:.3f} ({std_B_cftb_norm[i]:.3f}) - RND A: {acc_A_cftb_rnd[i]:.3f} ({std_A_cftb_rnd[i]:.3f}) B: {acc_B_cftb_rnd[i]:.3f} ({std_B_cftb_rnd[i]:.3f})")
    print()

    cum_A_fta += acc_A_fta/len(res.keys())
    cum_B_fta += acc_B_fta/len(res.keys())
    cum_A_ftb += acc_A_ftb/len(res.keys())
    cum_B_ftb += acc_B_ftb/len(res.keys())
    cum_A_cftb_raw += acc_A_cftb_raw[-1]/len(res.keys())
    cum_B_cftb_raw += acc_B_cftb_raw[-1]/len(res.keys())
    cum_A_cftb_norm += acc_A_cftb_norm[-1]/len(res.keys())
    cum_B_cftb_norm += acc_B_cftb_norm[-1]/len(res.keys())
    cum_A_cftb_rnd += acc_A_cftb_rnd[-1]/len(res.keys())
    cum_B_cftb_rnd += acc_B_cftb_rnd[-1]/len(res.keys())

print(f"AVG STATISTICS")

print(f"FTA -- A: {cum_A_fta:.3f} B: {cum_B_fta:.3f}")
print(f"FTB -- A: {cum_A_ftb:.3f} B: {cum_B_ftb:.3f}")
print(f"CFTB -- RAW A: {cum_A_cftb_raw:.3f} B: {cum_B_cftb_raw:.3f} - NORM A: {cum_A_cftb_norm:.3f} B: {cum_B_cftb_norm:.3f} - RND A: {cum_A_cftb_rnd:.3f} B: {cum_B_cftb_rnd:.3f}")


