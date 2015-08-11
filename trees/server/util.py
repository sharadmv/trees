
labels_template = "<div class='tree-label'>%s</div>"
label_template = "<div class='tree-label-text'>%s</div>"
def convert_labels_to_html(labels):
    labels = [label_template % s for s in labels]
    return labels_template % (''.join(labels))
