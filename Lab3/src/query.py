# This file is part of Lerot.
#
# Lerot is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Lerot is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Lerot.  If not, see <http://www.gnu.org/licenses/>.

"""
Interface to query data with functionality for reading queries from svmlight
format, both sequentially and in batch mode.
"""

import sys
import gc
import gzip
import numpy as np
import os.path
from document import Document


class SimpleBufferedLineReader:
    """Read lines from a file, but keep a short buffer to allow rewinds"""

    __prev__ = None
    __nextl_ = None
    __fh__ = None

    def __init__(self, fh):
        self.__fh__ = fh

    def has_next(self):
        if self.__nextl_ == None:  # not set
            self.__nextl_ = self.__fh__.readline()
        if len(self.__nextl_) < 1:  # empty string - EOF
            return False
        return True

    def __iter__(self):
        return self

    def next(self):
        if self.__nextl_ == None:  # not set
            self.__nextl_ = self.__fh__.readline()
        if len(self.__nextl_) < 1:  # empty string - EOF
            raise StopIteration
        self.__prev__ = self.__nextl_
        self.__nextl_ = None
        return self.__prev__

    __next__ = next

    def rewind(self):
        if self.__prev__:
            self.__nextl_ = self.__prev__
            self.__prev__ = None


class Query:
    __qid__ = None
    __feature_vectors__ = None
    __labels__ = None
    __predictions__ = None
    __comments__ = None
    # document ids will be initialized as zero-based, os they can be used to
    # retrieve labels, predictions, and feature vectors
    __docids__ = None
    __ideal__ = None

    def __hash__(self):
        return self.__qid__.__hash__()

    def __eq__(self, other):
        return (isinstance(other, self.__class__)
                and self.__qid__ == other.__qid__)

    def __init__(self, qid, feature_vectors, labels=None, comments=None):
        self.__qid__ = qid
        self.__feature_vectors__ = feature_vectors
        self.__labels__ = np.asarray(labels, dtype="float32")
        self.__docids__ = [Document(x) for x in range(len(labels))]
        self.__comments__ = comments

    def has_ideal(self):
        return not self.__ideal__ is None

    def set_ideal(self, ideal):
        self.__ideal__ = ideal

    def get_ideal(self):
        return self.__ideal__

    def get_qid(self):
        return self.__qid__

    def get_docids(self):
        return self.__docids__

    def get_document_count(self):
        return len(self.__docids__)

    def get_feature_vectors(self):
        return self.__feature_vectors__

    def set_feature_vector(self, docid, feature_vector):
        self.__feature_vectors__[docid.get_id()] = feature_vector

    def get_feature_vector(self, docid):
        return self.__feature_vectors__[docid.get_id()]

    def get_labels(self):
        return self.__labels__

    def set_label(self, docid, label):
        self.__labels__[docid.get_id()] = label

    def get_label(self, docid):
        return self.__labels__[docid.get_id()]

    def get_labelById(self, docid):
        return self.__labels__[docid]

    def set_labels(self, labels):
        self.__labels__ = labels

    def get_comments(self):
        return self.__comments__

    def get_comment(self, docid):
        if self.__comments__ is not None:
            return self.__comments__[docid.get_id()]
        return None

    def get_predictions(self):
        return self.__predictions__

    def get_prediction(self, docid):
        if self.__predictions__:
            return self.__predictions__[docid.get_id()]
        return None

    def set_predictions(self, predictions):
        self.__predictions__ = predictions

    def write_to(self, fh, sparse=False):
        for doc in self.__docids__:
            features = [':'.join((repr(pos + 1),
                                  repr(value))) for pos, value in enumerate(
                self.get_feature_vector(doc)) if not (value == 0 and sparse)]
            print >> fh, self.get_label(doc), ':'.join(("qid",
                                                        self.get_qid())), ' '.join(features),
            comment = self.get_comment(doc)
            if comment:
                print >> fh, comment
            else:
                print >> fh, ""


class QueryStream:
    """iterate over a stream of queries, only keeping one query at a time"""
    __reader__ = None
    __numFeatures__ = 0

    def __init__(self, fh, num_features, preserve_comments=False):
        self.__reader__ = SimpleBufferedLineReader(fh)
        self.__num_features__ = num_features
        self.__preserve_comments__ = preserve_comments
        self.__fh__ = fh

    def __iter__(self):
        return self

    # with some inspiration from multiclass.py
    # http://svmlight.joachims.org/svm_struct.html
    # takes self and number of expected features
    # returns qid and features, one query at a time
    def next(self):
        prev = None
        instances = None
        targets = None
        comments = None
        initialized = False
        for line in self.__reader__:
            line = line.rstrip("\n")
            # explicitly start a new query
            if line.startswith("# qid "):
                # if we have already read a query - break
                # don't need to rewind, we'll just skip the comment next time
                if initialized:
                    break
                # otherwise just start reading
                else:
                    continue
            # remove comments
            if line.startswith("#"):
                continue
            comment = ""
            if line.find("#") >= 0:
                comment_index = line.find("#")
                comment = line[comment_index:]
                line = line[:comment_index]
            # extract target and features
            tokens = line.split()
            if not tokens:
                continue
            target = float(tokens[0])
            qid = tokens[1].split(':')[1]

            if not qid:
                print >> sys.stderr, "Invalid line - skipping '", line, "'"
                continue
            if not prev:
                prev = qid
            if qid != prev:
                self.__reader__.rewind()
                break
            features = [tuple(t.split(":")) for t in tokens[2:]]
            tmp_array = np.zeros((1, self.__num_features__), dtype="float32")
            for k, v in features:
                tmp_array[0, int(k) - 1] = float(v)
            if not initialized:
                instances = tmp_array
                targets = np.array([target], dtype="float32")
                if self.__preserve_comments__:
                    comments = [comment]
                initialized = True
            else:
                instances = np.vstack((instances, tmp_array))
                targets = np.hstack((targets, target))
                if self.__preserve_comments__:
                    comments.append(comment)

        if not initialized:
            raise StopIteration
        else:
            return Query(prev, instances, targets, comments)

    __next__ = next  # Python 3.X compatibility

    # read all queries from a file at once
    def read_all(self):
        queries = {}
        index = 0
        for query in self:
            #if index % 100 == 0:
                #print("loaded ", index)
            index += 1
            queries[query.get_qid()] = query
        return queries


class Queries:
    """a list of queries with some convenience functions"""

    __num_features__ = 0
    __queries__ = None

    # cache immutable query values
    __qids__ = None
    __feature_vectors__ = None
    __labels__ = None

    def __init__(self, fh, num_features, preserve_comments=False):
        self.__queries__ = QueryStream(fh, num_features,
                                       preserve_comments).read_all()
        self.__num_features__ = num_features

    def __iter__(self):
        return iter(list(self.__queries__.values()))

    def __getitem__(self, index):
        return self.get_query(index)

    def __len__(self):
        return len(self.__queries__)

    def keys(self):
        return self.__queries__.keys()

    def values(self):
        return self.__queries__.values()

    def get_query(self, index):
        return self.__queries__[index]

    def get_qids(self):
        if not self.__qids__:
            self.__qids__ = [query.get_qid() for query in self]
        return self.__qids__

    def get_labels(self):
        if not self.__labels__:
            self.__labels__ = [query.get_labels() for query in self]
        return self.__labels__

    def get_feature_vectors(self):
        if not self.__feature_vectors__:
            self.__feature_vectors__ = [query.get_feature_vectors()
                                        for query in self]
        return self.__feature_vectors__

    def set_predictions(self):
        raise NotImplementedError("Not yet implemented")

    def get_predictions(self):
        if not self.__predictions__:
            self.__predictions__ = [query.get_predictions() for query in self]
        return self.__predictions__

    def get_size(self):
        return self.__len__()


def load_queries(filename, features, preserve_comments=False):
    """Utility method for loading queries from a file."""
    # try:
    #     queries = cPickle.load(open( filename + "_pickle", "rb" ))
    #     print "loaded queries from pickle ",filename + "_pickle"
    # except:
    if filename.endswith(".gz"):
        fh = gzip.open(filename)
    else:
        fh = open(filename)
    gc.disable()
    queries = Queries(fh, features, preserve_comments)
    gc.enable()
    fh.close()
    # cPickle.dump(queries, open( filename + "_pickle", "w" ))
    #print("loaded queries from file")
    return queries