# -*- coding: utf-8 -*-
# Author: Douglas Creager <dcreager@dcreager.net>
# This file is placed into the public domain.

# Calculates the current version number.  If possible, this is the
# output of “git describe”, modified to conform to the versioning
# scheme that setuptools uses.  If “git describe” returns an error
# (most likely because we're in an unpacked copy of a release tarball,
# rather than in a git working copy), then we fall back on reading the
# contents of the RELEASE-VERSION file.
#
# To use this script, simply import it your setup.py file, and use the
# results of get_git_version() as your package version:
#
# from version import *
#
# setup(
#     version=get_git_version(),
#     .
#     .
#     .
# )
#
# This will automatically update the RELEASE-VERSION file, if
# necessary.  Note that the RELEASE-VERSION file should *not* be
# checked into git; please add it to your top-level .gitignore file.
#
# You'll probably want to distribute the RELEASE-VERSION file in your
# sdist tarballs; to do this, just create a MANIFEST.in file that
# contains the following line:
#
#   include RELEASE-VERSION

__all__ = ("get_git_version")

import os
from subprocess import Popen, PIPE

_refname = '$Format: %D$'
_tree_hash = '$Format: %t$'
_commit_info = '$Format:%cd by %aN$'
_commit_hash = '$Format: %h$'

def call_git_describe(abbrev=4):

    try:
        dirname = os.path.abspath(os.path.dirname(__file__))
        p = Popen(['git', 'describe', '--abbrev=%d' % abbrev, '--dirty'],
                  stdout=PIPE, stderr=PIPE,
                  cwd=os.path.join('..',dirname))
        p.stderr.close()
        line = p.stdout.readlines()[0]
        return line.strip()

    except:
        return None

def read_release_keywords(keyword):

    refnames = keyword.strip()
    if refnames.startswith("$Format"): return None

    refs = set([r.strip() for r in refnames.strip("()").split(",")])
    TAG = "tag: "
    tags = set([r[len(TAG):] for r in refs if r.startswith(TAG)])
    if not tags: return None
    return sorted(tags)[-1]
        
def read_release_version():

    import re
    dirname = os.path.abspath(os.path.dirname(__file__))


    try:

        f = open(os.path.join(dirname,"_version.py"), "r")
        for line in f.readlines():

            m = re.match("__version__ = '([^']+)'", line)
            if m:
                ver = m.group(1)
                return ver
#        try:
#            version = f.readlines()[0]
#            return version.strip()
#        finally:
#            f.close()

    except:
        return None

    return None

def write_release_version(version):

    dirname = os.path.abspath(os.path.dirname(__file__))
    f = open(os.path.join(dirname,"_version.py"), "w")
    f.write("__version__ = '%s'\n" % version)
    f.close()

def get_git_version(abbrev=4):

    # Read in the version that's currently in _version.py.
    release_version = read_release_version()

    # First try to get the current version using “git describe”.
    git_version = call_git_describe(abbrev)

    # Try to deduce the version from keyword expansion
    keyword_version = read_release_keywords(_refname)

    # If that doesn't work, fall back on the value that's in
    # _version.py.
    if git_version is not None:
        version = git_version
    elif release_version is not None:
        version = release_version
    elif keyword_version is not None:
        version = keyword_version
    else:
        version = 'unknown'

    # If we still don't have anything, that's an error.
    if version is None:
        raise ValueError("Cannot find the version number!")

    # If the current version is different from what's in the
    # _version.py file, update the file to be current.
    if version != release_version and version != 'unknown':
        write_release_version(version)

    # Finally, return the current version.
    return version

if __name__ == "__main__":
    print(get_git_version())
