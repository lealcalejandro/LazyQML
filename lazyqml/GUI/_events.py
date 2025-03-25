from _widgets import *
from _generator import generate_code

# Enable/disable widgets depending on CV/bagging options
def on_change_cv(change):
    splits_widget.disabled = not change['new']
    repeats_widget.disabled = not change['new']

def on_change_bag(change):
    nsamples_widget.disabled = not change['new']
    nfeatures_widget.disabled = not change['new']
    nestimators_widget.disabled = not change['new']

# Automatically update textbox
def on_change_params(change):
    generate_code(None)

# Trigger functions on events
def set_events():
    cv_checkbox.observe(on_change_cv, names='value')
    qnn_bag_checkbox.observe(on_change_bag, names='value')

    generate_code_button.on_click(generate_code)

    for w in all_widgets:
        w.observe(on_change_params)