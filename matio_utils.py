import csv
import time
import os
from materials_io.utils.interface import get_available_parsers, get_parser
import multiprocessing as mp


def run_parser(file_parser):
    """Helper function for multiprocessing.

    Parameters:
    file_parser (parser, file): 2-tuple containing a parser name
    to use for parsing and a file(s) to parse.

    Returns:
    metadata {file_name: {parser_name: metadata_dict}}: Returns
    dictionary of metadata with parser and file names included.
    """
    parser = get_parser(file_parser[0])
    file = file_parser[1]

    try:
        metadata = {file: {file_parser[0]: parser.parse(file)}}
        return metadata
    except Exception as e:
        return None


def run_all_parsers_mp(directory, exclude_parsers=None, processes=mp.cpu_count()):
    """Runs all parsers on a directory but uses multiprocessing.

    Parameters:
    directory (str): Directory to run parsers on.
    exclude_parsers (list): List of parsers to not run.

    Returns:
    file_metadata (file_name: {parser_name: metadata_dict}}):
    List of dictionaries of metadata as returned by run_parser.
    """
    start_time = time.time()
    parsers = get_available_parsers()
    file_metadata = []
    task_queue = []

    if exclude_parsers is not None:
        parsers = list(set(parsers.keys()).difference(exclude_parsers))

    print("starting...")

    for parser in parsers:
        parser_obj = get_parser(parser)

        for root, dirs, files in os.walk(directory):
            # Generate the full paths
            dirs = [os.path.join(root, d) for d in dirs]
            files = [os.path.join(root, f) for f in files]

            for group in parser_obj.group(files, dirs):
                task_queue.append((parser, group))

    print("It took {} seconds to generate the queue. {} jobs in queue".format(time.time() - start_time,
                                                                              len(task_queue)))
    print("starting job processing...")

    pools = mp.Pool(processes)

    for metadata in pools.imap_unordered(run_parser, task_queue):
        file_metadata.append(metadata)
        if (len(file_metadata) % 1000) == 0:
            print("{} out of {} files processed".format(len(file_metadata), len(task_queue)))
            print("{} seconds have passed\n".format(time.time() - start_time))

    pools.close()
    pools.join()

    file_metadata = [metadata for metadata in file_metadata if metadata is not None]

    print("Finished in {} seconds".format(time.time() - start_time))
    print("{} number of metadata".format(len(file_metadata)))
    return file_metadata


def matio_label_gen(directory, label_file=None, exclude_parsers=None):
    """Generates file metadata using run_all_parsers_mp
    and then writes file names and parser names to a .csv for
    successfully extracted metadata.

    Parameters:
    directory (str): Directory of files to write labels for.
    label_file (str): Name of .csv to write labels to.
    exclude_parsers (list): List of parsers to not run.
    """
    file_row = []

    if label_file is None:
        label_file = os.path.basename(directory) + ".csv"

    file_metadata = run_all_parsers_mp(directory, exclude_parsers=exclude_parsers)

    for metadata in file_metadata:
        file_path = list(metadata.keys())[0]
        file_label = list(metadata[file_path].keys())[0]

        if isinstance(file_path, list):
            for path in file_path:
                file_row.append([path, os.path.getsize(path), file_label])
        else:
            file_row.append([file_path[0], os.path.getsize(file_path[0]), file_label])

    with open(label_file, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["path", "size", "file_label"])

        for row in file_row:
            csv_writer.writerow(row)

    print("Done writing labels")
