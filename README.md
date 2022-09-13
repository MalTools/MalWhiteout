# MalWhiteout


Please make sure that you have installed the required python libraries, e.g., cleanlab, sklearn, numpy, etc.

Please note that currently our tool uses version 1.0 of the cleanlab package, while it has now been upgraded to 2.0. The [cleanlab](https://github.com/cleanlab/cleanlab) project provides detailed documentations on how to use it as well as how to migrate to v2.0 from v1.0. Don't hesitate to read them if you might want to work with the latest new v2.0.

Our tool currently integrated three open source malware detection approaches: [Drebin](https://github.com/MLDroid/drebin), [CSBD](https://github.com/MLDroid/csbd), [MalScan](https://github.com/malscan-android/MalScan). You need to first extract features seperately using each approach, i.e., .data file for Drebin, .txt file for CSBD and .csv file for MalScan. Then feed the features along with labels in our tool ().

After obtaining the (out-of-sample) predicted probabilities of each individual model, use the ensemble learning method to find label errors using cleanlab ().


Extension:

Besides the used three approaches (i.e., Drebin, CSBD, MalScan), you can also try to add (or replace with) other malware detection techniques. But keep in mind that each individual model should perform not badly in malware detection so that the ensembled result would be better.

