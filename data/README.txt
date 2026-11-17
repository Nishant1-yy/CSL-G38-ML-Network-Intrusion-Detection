Default dataset: NSL-KDD
========================

The original project instructions already identified NSL-KDD. This project
keeps that dataset as its default and does not bundle or fabricate records.

Source information
------------------

Official University of New Brunswick dataset page:
https://www.unb.ca/cic/datasets/nsl.html

The project’s previous data note also identified this mirror of the NSL-KDD
files:
https://github.com/defcom17/NSL_KDD

Direct file URLs listed in the previous project note:
https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt
https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt

Before downloading, using, or redistributing records, review the source page
and repository terms. No separate dataset license is asserted by this project.
When using the data, cite the source and the dataset paper:

M. Tavallaee, E. Bagheri, W. Lu, and A. Ghorbani, “A Detailed Analysis of the
KDD CUP 99 Data Set,” IEEE Symposium on Computational Intelligence for
Security and Defense Applications, 2009.

Setup
-----

Create data/raw/ and place KDDTrain+.txt there. Optionally add the official
KDDTest+.txt alongside it; the training workflow reserves that file for final
evaluation. NSL-KDD is headerless and is parsed using its standard 41 feature
names, label, and difficulty column. Difficulty is metadata, not a predictor.

Labeled CSV datasets are also supported under data/raw/. CSV files must include
a target column named Label, class, attack, or target. The corresponding source,
citation, usage terms, and feature definitions should be recorded by the user.

Downloaded files are excluded from version control by the repository
.gitignore. Do not commit dataset records or credentials.
