
#print ("---Temp Setup----")
#import sys
#delete_path ='/glade/scratch/hn1174/multi_project/baseline/FILTER/src'
#print ("ori path", sys.path)
#if (delete_path in sys.path):
#	print ("Delete path")
#	sys.path.remove(delete_path)


from setuptools import setup, find_packages
setup(name='project', version='1.0', package_dir={"":"code"}, packages=find_packages(where='code'))

