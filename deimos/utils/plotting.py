'''
A minimal set of useful plotting tools

Tom Stuttard
'''

import matplotlib
import matplotlib.pyplot as plt

import os

def dump_figures_to_pdf(file_path, figures=None):
    """
    Save figures into a single PDF file
    Can either provide a list of figures, or if not then all in the current session will be saved
    """

    # Use a list of figures if one is provided by user
    # Otherwise grab all figures from the global list stored by matpotlib
    if figures is not None:
        fig_nums = [
            f.fig.number if isinstance(f, Figure) else f.number for f in figures
        ]  # Handle my Figure class vs matplotligb figure class
    else:
        fig_nums = plt.get_fignums()

    # Check inputs
    file_path = os.path.abspath(file_path)
    parent_dir = os.path.dirname(file_path)
    assert os.path.isdir(parent_dir), 'Could not dump figure to PDF : Directory "%s" does not exist' % parent_dir

    # Check found any figures
    if len(fig_nums) > 0:

        # Create the PDF
        with matplotlib.backends.backend_pdf.PdfPages(file_path) as pdf:

            # Write figures to it
            for fig_num in fig_nums:
                pdf.savefig(fig_num) #TODO option to set dpi

        # Report
        print( "Dumped %i figures to PDF : %s" % (len(fig_nums), file_path) )

    else:
        # Complain
        print("No figures found, could not dump to PDF")

