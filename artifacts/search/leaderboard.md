# A4 search leaderboard (VALIDATION F1 only, test never used)

Updated 2026-11-07T04:18. Budget 30 trials/arm, stop_at 2026-11-23T23:59:00+07:00.

Target: beat `b5_logreg` val F1 = 0.1696 by margin = seed_std.

| arm | trials done | failed | best seed-0 val F1 | proposed config (top by 3-seed mean) | 3-seed val F1 | beats target |
|---|---|---|---|---|---|---|
| at_pre | 12 | 0 | 0.1945 | (confirm phase pending) |  |  |
| at_scratch | 24 | 0 | 0.1897 | (confirm phase pending) |  |  |
| cnnbilstm | 14 | 0 | 0.1782 | (confirm phase pending) |  |  |

## at_pre: top 10 trials

| trial | val F1 | params |
|---|---|---|
| 1 | 0.1945 | `{'train.lr': 0.0005495716186411617, 'train.weight_decay': 0.08626905220714909, 'train.batch_size': 32, 'train.patience': 10, 'train.pos_weight_scale': 1.062448776694192, 'train.label_smoothing': 0.04146619399905236, 'dropout': 0.07936668363138809, 'train.finetune_mode': 'full', 'pretrain': 'e10'}` |
| 10 | 0.1890 | `{'train.lr': 0.0007472497786533267, 'train.weight_decay': 0.07557470169456365, 'train.batch_size': 32, 'train.patience': 10, 'train.pos_weight_scale': 1.6488384737616495, 'train.label_smoothing': 0.05780226889296412, 'dropout': 0.07897505870130689, 'train.finetune_mode': 'full', 'pretrain': 'e10'}` |
| 9 | 0.1864 | `{'train.lr': 0.0003867189079923129, 'train.weight_decay': 0.0004696497161394397, 'train.batch_size': 64, 'train.patience': 6, 'train.pos_weight_scale': 0.4834944671543633, 'train.label_smoothing': 0.09342139979247938, 'dropout': 0.1841897867897688, 'train.finetune_mode': 'layerwise', 'train.layerwise_decay': 0.6403752479658208, 'pretrain': 'e10_lr3e-4'}` |
| 2 | 0.1831 | `{'train.lr': 0.0007717846862118063, 'train.weight_decay': 0.011103525577197588, 'train.batch_size': 128, 'train.patience': 6, 'train.pos_weight_scale': 0.3574232010219508, 'train.label_smoothing': 0.03154283509241839, 'dropout': 0.10911323128278679, 'train.finetune_mode': 'layerwise', 'train.layerwise_decay': 0.5459201648366127, 'pretrain': 'e20'}` |
| 11 | 0.1819 | `{'train.lr': 0.0005308223874624805, 'train.weight_decay': 0.04029349895075665, 'train.batch_size': 32, 'train.patience': 10, 'train.pos_weight_scale': 3.6193786271497177, 'train.label_smoothing': 0.05076290760856234, 'dropout': 0.14903297936847365, 'train.finetune_mode': 'full', 'pretrain': 'e10'}` |
| 3 | 0.1747 | `{'train.lr': 8.562913278818231e-05, 'train.weight_decay': 0.0005410990556767879, 'train.batch_size': 128, 'train.patience': 10, 'train.pos_weight_scale': 0.3272363916738979, 'train.label_smoothing': 0.0837944907498804, 'dropout': 0.02882952236818892, 'train.finetune_mode': 'layerwise', 'train.layerwise_decay': 0.7721804838852706, 'pretrain': 'canonical'}` |
| 8 | 0.1719 | `{'train.lr': 0.0004966111413793923, 'train.weight_decay': 0.026421463889976064, 'train.batch_size': 256, 'train.patience': 10, 'train.pos_weight_scale': 3.7375322294860673, 'train.label_smoothing': 0.0855803342392611, 'dropout': 0.0035142252555005913, 'train.finetune_mode': 'frozen', 'pretrain': 'canonical'}` |
| 6 | 0.1671 | `{'train.lr': 0.0005783708472141561, 'train.weight_decay': 0.005543941231681782, 'train.batch_size': 32, 'train.patience': 4, 'train.pos_weight_scale': 1.3431057091018288, 'train.label_smoothing': 0.0019193198309333526, 'dropout': 0.0904724450023648, 'train.finetune_mode': 'full', 'pretrain': 'e10_lr3e-4'}` |
| 7 | 0.1671 | `{'train.lr': 0.0001519656816607236, 'train.weight_decay': 0.0052841574123838175, 'train.batch_size': 256, 'train.patience': 10, 'train.pos_weight_scale': 2.337191513989615, 'train.label_smoothing': 0.07038885835403663, 'dropout': 0.030068066193690332, 'train.finetune_mode': 'layerwise', 'train.layerwise_decay': 0.5672517370960972, 'pretrain': 'canonical'}` |
| 5 | 0.1641 | `{'train.lr': 1.0970213817508292e-05, 'train.weight_decay': 0.030677508679378802, 'train.batch_size': 256, 'train.patience': 4, 'train.pos_weight_scale': 1.2907125376089443, 'train.label_smoothing': 0.05722519057908734, 'dropout': 0.06692448979218549, 'train.finetune_mode': 'full', 'pretrain': 'e20'}` |

## at_scratch: top 10 trials

| trial | val F1 | params |
|---|---|---|
| 13 | 0.1897 | `{'train.lr': 4.7446855437968263e-05, 'train.weight_decay': 0.08909944882166042, 'train.batch_size': 64, 'train.patience': 10, 'train.pos_weight_scale': 2.310010424384453, 'train.label_smoothing': 0.046435191239550186, 'dropout': 0.237079276275306}` |
| 21 | 0.1889 | `{'train.lr': 2.4595196042086015e-05, 'train.weight_decay': 0.08344314613640477, 'train.batch_size': 64, 'train.patience': 10, 'train.pos_weight_scale': 1.9614453623917354, 'train.label_smoothing': 0.07418575131334817, 'dropout': 0.08804315772666543}` |
| 7 | 0.1885 | `{'train.lr': 3.4662172842139684e-05, 'train.weight_decay': 0.04871454545414109, 'train.batch_size': 64, 'train.patience': 6, 'train.pos_weight_scale': 1.2427692563092039, 'train.label_smoothing': 0.04081368027612812, 'dropout': 0.07110809407290831}` |
| 16 | 0.1885 | `{'train.lr': 3.548113519247008e-05, 'train.weight_decay': 0.06623783692338954, 'train.batch_size': 64, 'train.patience': 10, 'train.pos_weight_scale': 1.9981610687452656, 'train.label_smoothing': 0.028437511133420974, 'dropout': 0.06554265399128234}` |
| 1 | 0.1875 | `{'train.lr': 2.56392017331631e-05, 'train.weight_decay': 0.043087600478391316, 'train.batch_size': 64, 'train.patience': 10, 'train.pos_weight_scale': 3.663052308384009, 'train.label_smoothing': 0.031342417815924284, 'dropout': 0.20769678470079422}` |
| 19 | 0.1874 | `{'train.lr': 5.348514742259482e-05, 'train.weight_decay': 0.03849763890596805, 'train.batch_size': 64, 'train.patience': 10, 'train.pos_weight_scale': 2.2080115230072535, 'train.label_smoothing': 0.02975107005824513, 'dropout': 0.14691180495685752}` |
| 0 | 0.1864 | `{'train.lr': 6.82407842709883e-05, 'train.weight_decay': 0.01448683388723935, 'train.batch_size': 64, 'train.patience': 10, 'train.pos_weight_scale': 1.1136276491758386, 'train.label_smoothing': 0.041919451440329485, 'dropout': 0.20556585011902784}` |
| 14 | 0.1864 | `{'train.lr': 4.845688501721154e-05, 'train.weight_decay': 0.05040816527041978, 'train.batch_size': 64, 'train.patience': 6, 'train.pos_weight_scale': 0.9675349051299799, 'train.label_smoothing': 0.015149794691120218, 'dropout': 0.07400417421920782}` |
| 15 | 0.1863 | `{'train.lr': 5.471532664978625e-05, 'train.weight_decay': 0.01919956156873423, 'train.batch_size': 64, 'train.patience': 6, 'train.pos_weight_scale': 2.002470008769155, 'train.label_smoothing': 0.01039251095790384, 'dropout': 0.2844166037438788}` |
| 22 | 0.1851 | `{'train.lr': 1.5575371550763645e-05, 'train.weight_decay': 0.07229773115988293, 'train.batch_size': 64, 'train.patience': 10, 'train.pos_weight_scale': 1.7992072945018427, 'train.label_smoothing': 0.054546193109025864, 'dropout': 0.016562899396265943}` |

## cnnbilstm: top 10 trials

| trial | val F1 | params |
|---|---|---|
| 3 | 0.1782 | `{'train.lr': 0.000386611625197214, 'train.weight_decay': 0.005495567374990245, 'train.batch_size': 128, 'train.patience': 4, 'train.pos_weight_scale': 0.8180062964429848, 'train.label_smoothing': 0.043674726302679906, 'dropout': 0.23296775549913007}` |
| 6 | 0.1773 | `{'train.lr': 3.3074211974723665e-05, 'train.weight_decay': 0.0014476702711160212, 'train.batch_size': 32, 'train.patience': 4, 'train.pos_weight_scale': 0.26835297130960717, 'train.label_smoothing': 0.06622020192683258, 'dropout': 0.11625702776198443}` |
| 4 | 0.1754 | `{'train.lr': 0.00011781710498405437, 'train.weight_decay': 0.07264850487520612, 'train.batch_size': 256, 'train.patience': 4, 'train.pos_weight_scale': 0.3011548676591997, 'train.label_smoothing': 0.09938520114212729, 'dropout': 0.29117409401315203}` |
| 0 | 0.1752 | `{'train.lr': 7.44714490408307e-05, 'train.weight_decay': 0.00011961308606153212, 'train.batch_size': 32, 'train.patience': 6, 'train.pos_weight_scale': 0.5238802939686513, 'train.label_smoothing': 0.062113383276929494, 'dropout': 0.15874262828311173}` |
| 9 | 0.1749 | `{'train.lr': 0.00015281611527743648, 'train.weight_decay': 0.0010932355063534291, 'train.batch_size': 64, 'train.patience': 6, 'train.pos_weight_scale': 0.40805871124174037, 'train.label_smoothing': 0.059125735265354584, 'dropout': 0.14677985009845773}` |
| 7 | 0.1741 | `{'train.lr': 9.866147353327376e-05, 'train.weight_decay': 0.0017567805401718767, 'train.batch_size': 128, 'train.patience': 10, 'train.pos_weight_scale': 1.5476181469425645, 'train.label_smoothing': 0.021463574639995365, 'dropout': 0.12502603205860113}` |
| 1 | 0.1733 | `{'train.lr': 1.8584885538668354e-05, 'train.weight_decay': 0.003473236559598501, 'train.batch_size': 128, 'train.patience': 4, 'train.pos_weight_scale': 0.2996075775780364, 'train.label_smoothing': 0.042812232759738944, 'dropout': 0.028959274698183768}` |
| 11 | 0.1733 | `{'train.lr': 0.0007445914989176157, 'train.weight_decay': 0.0005544936016622726, 'train.batch_size': 128, 'train.patience': 4, 'train.pos_weight_scale': 0.3373688489188721, 'train.label_smoothing': 0.054850534666393086, 'dropout': 0.05405764397578399}` |
| 5 | 0.1724 | `{'train.lr': 0.000398581100804794, 'train.weight_decay': 0.006389271811451123, 'train.batch_size': 32, 'train.patience': 10, 'train.pos_weight_scale': 0.8499322069468666, 'train.label_smoothing': 0.050400043937915265, 'dropout': 0.09706239526040965}` |
| 8 | 0.1723 | `{'train.lr': 0.0001939473581738875, 'train.weight_decay': 0.009648148518478433, 'train.batch_size': 64, 'train.patience': 4, 'train.pos_weight_scale': 1.1276981248887328, 'train.label_smoothing': 0.02208379150614619, 'dropout': 0.27553759397688016}` |
