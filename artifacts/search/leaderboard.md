# A4 search leaderboard (VALIDATION F1 only, test never used)

Updated 2026-11-06T18:43. Budget 30 trials/arm, stop_at 2026-11-23T23:59:00+07:00.

Target: beat `b5_logreg` val F1 = 0.1696 by margin = seed_std.

| arm | trials done | failed | best seed-0 val F1 | proposed config (top by 3-seed mean) | 3-seed val F1 | beats target |
|---|---|---|---|---|---|---|
| at_pre | 5 | 0 | 0.1945 | (confirm phase pending) |  |  |
| at_scratch | 4 | 0 | 0.1875 | (confirm phase pending) |  |  |
| cnnbilstm | 4 | 0 | 0.1782 | (confirm phase pending) |  |  |

## at_pre: top 10 trials

| trial | val F1 | params |
|---|---|---|
| 1 | 0.1945 | `{'train.lr': 0.0005495716186411617, 'train.weight_decay': 0.08626905220714909, 'train.batch_size': 32, 'train.patience': 10, 'train.pos_weight_scale': 1.062448776694192, 'train.label_smoothing': 0.04146619399905236, 'dropout': 0.07936668363138809, 'train.finetune_mode': 'full', 'pretrain': 'e10'}` |
| 2 | 0.1831 | `{'train.lr': 0.0007717846862118063, 'train.weight_decay': 0.011103525577197588, 'train.batch_size': 128, 'train.patience': 6, 'train.pos_weight_scale': 0.3574232010219508, 'train.label_smoothing': 0.03154283509241839, 'dropout': 0.10911323128278679, 'train.finetune_mode': 'layerwise', 'train.layerwise_decay': 0.5459201648366127, 'pretrain': 'e20'}` |
| 3 | 0.1747 | `{'train.lr': 8.562913278818231e-05, 'train.weight_decay': 0.0005410990556767879, 'train.batch_size': 128, 'train.patience': 10, 'train.pos_weight_scale': 0.3272363916738979, 'train.label_smoothing': 0.0837944907498804, 'dropout': 0.02882952236818892, 'train.finetune_mode': 'layerwise', 'train.layerwise_decay': 0.7721804838852706, 'pretrain': 'canonical'}` |
| 0 | 0.1611 | `{'train.lr': 0.00012520653814999459, 'train.weight_decay': 0.013981961408994055, 'train.batch_size': 256, 'train.patience': 10, 'train.pos_weight_scale': 0.7238517082858716, 'train.label_smoothing': 0.07917250380826646, 'dropout': 0.15866847592587133, 'train.finetune_mode': 'frozen', 'pretrain': 'e20'}` |
| 4 | 0.1446 | `{'train.lr': 3.9109331722510566e-05, 'train.weight_decay': 0.00022708223335487552, 'train.batch_size': 256, 'train.patience': 4, 'train.pos_weight_scale': 0.32438118414908174, 'train.label_smoothing': 0.05759464955561793, 'dropout': 0.27878885927286423, 'train.finetune_mode': 'frozen', 'pretrain': 'canonical'}` |

## at_scratch: top 10 trials

| trial | val F1 | params |
|---|---|---|
| 1 | 0.1875 | `{'train.lr': 2.56392017331631e-05, 'train.weight_decay': 0.043087600478391316, 'train.batch_size': 64, 'train.patience': 10, 'train.pos_weight_scale': 3.663052308384009, 'train.label_smoothing': 0.031342417815924284, 'dropout': 0.20769678470079422}` |
| 0 | 0.1864 | `{'train.lr': 6.82407842709883e-05, 'train.weight_decay': 0.01448683388723935, 'train.batch_size': 64, 'train.patience': 10, 'train.pos_weight_scale': 1.1136276491758386, 'train.label_smoothing': 0.041919451440329485, 'dropout': 0.20556585011902784}` |
| 3 | 0.1773 | `{'train.lr': 0.00023604883174756995, 'train.weight_decay': 0.03190634197513423, 'train.batch_size': 128, 'train.patience': 6, 'train.pos_weight_scale': 0.865481030772183, 'train.label_smoothing': 0.09085955030930956, 'dropout': 0.08808424451210385}` |
| 2 | 0.1748 | `{'train.lr': 0.0005659503131926907, 'train.weight_decay': 0.04828586218554464, 'train.batch_size': 256, 'train.patience': 10, 'train.pos_weight_scale': 1.0963140561774434, 'train.label_smoothing': 0.06918771139504734, 'dropout': 0.09465468930181888}` |

## cnnbilstm: top 10 trials

| trial | val F1 | params |
|---|---|---|
| 3 | 0.1782 | `{'train.lr': 0.000386611625197214, 'train.weight_decay': 0.005495567374990245, 'train.batch_size': 128, 'train.patience': 4, 'train.pos_weight_scale': 0.8180062964429848, 'train.label_smoothing': 0.043674726302679906, 'dropout': 0.23296775549913007}` |
| 0 | 0.1752 | `{'train.lr': 7.44714490408307e-05, 'train.weight_decay': 0.00011961308606153212, 'train.batch_size': 32, 'train.patience': 6, 'train.pos_weight_scale': 0.5238802939686513, 'train.label_smoothing': 0.062113383276929494, 'dropout': 0.15874262828311173}` |
| 1 | 0.1733 | `{'train.lr': 1.8584885538668354e-05, 'train.weight_decay': 0.003473236559598501, 'train.batch_size': 128, 'train.patience': 4, 'train.pos_weight_scale': 0.2996075775780364, 'train.label_smoothing': 0.042812232759738944, 'dropout': 0.028959274698183768}` |
| 2 | 0.1652 | `{'train.lr': 1.7960562886891067e-05, 'train.weight_decay': 0.006169300530668235, 'train.batch_size': 256, 'train.patience': 10, 'train.pos_weight_scale': 0.9541442791000934, 'train.label_smoothing': 0.05052367200185492, 'dropout': 0.11606779533556778}` |
