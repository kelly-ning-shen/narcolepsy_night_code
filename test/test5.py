signal_labels = ['C3','C4']
channels = ['C3','C4','O1']
channels_used = dict.fromkeys(channels)
for ch in channels_used:
    print(ch)
    # if ch in signal_labels:
    #     print(f'in: {ch}')
    # else:
    #     del channels_used[ch]
    #     print(f'remove {ch}')
print(channels_used)

# Wrong!