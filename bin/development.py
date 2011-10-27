from analysis.process_file import process_file
from analysis.process_flight import process_flight

def main():
    file_path = os.path.join('.', 'file.csv')
    segments = process_file(file_path, param_group='FFD',split=False)
    # process one?
    
    process_flight(segments[0])
    graph_flight(segments[0])
    
    
if __name__ == '__main__':
    main()
    