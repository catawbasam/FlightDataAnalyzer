from analysis.process_file import process_file
from analysis.process_flight import process_flight

def main():
    file_path = os.path.join('.', 'file.csv')
    hdf = process_file(file_path)
    process_flight(hdf)
    
    
if __name__ == '__main__':
    main()
    