import sys
from notmuch import Database, Query
from progressbar import ProgressBar, Percentage, Bar, ETA

def erase_irrelevant_tags(lst):
    irrelevant = {"attachment", "draft", "encrypted", "new", "signed", "unread",
                  "inbox", "replied", "flagged", "deleted"}
    return filter(lambda x: x not in irrelevant, lst)

def get_training_data(progress=False):
    training_data = []
    training_labels = []
    db = Database()
    # query that returns all the messages
    q = Query(db, '')
    if progress:
        count = q.count_messages()
        n = 0
        pbar = ProgressBar(widgets=[Percentage(), Bar(), ETA()],
                           maxval=count).start()

    data = []
    for m in q.search_messages():
        if progress:
            n += 1
            pbar.update(n)

        data.append(m.get_header('To'))
        data.append(m.get_header('From'))
        data.append(m.get_header('Subject'))
        data.append(m.get_part(1).decode("utf8", errors="ignore"))
        try:
            training_data.append('\n'.join(data))
        except UnicodeDecodeError:
            print(map(lambda x: type(x), data))
            sys.exit(1)
        training_labels.append(erase_irrelevant_tags(list(m.get_tags())))
        data = []

    if progress:
        pbar.finish()
    return training_data, training_labels

def get_new_mails():
    db = Database()
    query = Query(db, 'tag:new')
    data = []
    ids = []
    m_data = []
    for m in query.search_messages():
        m_data.append(m.get_header('To'))
        m_data.append(m.get_header('From'))
        m_data.append(m.get_header('Subject'))
        m_data.append(m.get_part(1).decode("utf8", errors="ignore"))
        try:
            data.append('\n'.join(m_data))
            ids.append(m.get_message_id())
        except UnicodeDecodeError:
            print(map(lambda x: type(x), m_data))
            sys.exit(1)
        m_data = []
    return data, ids

def write_tags(ids, tags):
    db = Database(mode=Database.MODE.READ_WRITE)
    for i, ts in zip(ids, tags):
        m = db.find_message(i)
        m.remove_tag("new")
        m.add_tag("inbox")
        for t in erase_irrelevant_tags(ts):
            m.add_tag(t)
