import numpy as np

import config


def get_range(ds, dlg, start):
    end = min(start + ds.max_n_utt, len(dlg))
    while True:
        try:
            data = ds.prepare_dialog(dlg[start:end], skip_links=True)
            return data, start, end
        except Exception as e:
            end -= 1


def predict_rec(ds, dlg, model, t):
    cres = np.zeros(shape=(len(dlg), len(config.emotion_categories)))
    lres = np.zeros(shape=(len(dlg), len(dlg)))
    last_end = -1
    for i in range(len(dlg)):
        data, start, end = get_range(ds, dlg, i)

        if end <= last_end:
            continue

        data = [t.unsqueeze(0) for t in data]
        utt_classes_logits, utt_links_logits = model(*data)

        cpred = utt_classes_logits.detach().cpu().numpy()[0]
        lpred = utt_links_logits.detach().cpu().numpy()[0]

        s = end - start
        cres[start:end] += cpred[:s]
        lres[start:end, start:end] += lpred[:s, :s]

        if end >= len(dlg):
            break

        last_end = end

    cpred = np.argmax(cres, axis=-1)
    lpred = (lres > t)
    return cpred, lpred


def predict(ds, dlg, model, t):
    try:
        data = ds.prepare_dialog(dlg, skip_links=True)
    except Exception as e:
        # traceback.print_exc()
        print(f"error occured, will be solved recursively: {e}")
        return predict_rec(ds, dlg, model, t)

    data = [t.unsqueeze(0) for t in data]
    utt_classes_logits, utt_links_logits = model(*data)

    cpred = utt_classes_logits.argmax(dim=-1).detach().cpu().flatten().numpy()
    lpred = (utt_links_logits > t).detach().cpu()[0].numpy()
    return cpred, lpred


