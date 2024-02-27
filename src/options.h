//
// Created by lxu on 23-6-26.
//

#ifndef MMCHAIN_ANALYSIS_SRC_OPTIONS_H
#define MMCHAIN_ANALYSIS_SRC_OPTIONS_H

#include <iostream>
#include <string>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <iomanip>
#include <fstream>

namespace po = boost::program_options;
namespace fs = boost::filesystem;

namespace DAT {

class Options {
public:
  static std::string log_directory;
  static bool print_to_screen;
  static int save_log_file;
  static bool store_whole_block;
  static long mem_size;
  static long seq_length;
  static long hid_size;
  static long head_num;
  static long head_blocksize;
  static long batch_size;
  static long batch_blocksize;
  static std::string dim_order_opt;
  static long compute_power;
  static bool enable_compute_utilization_constraint;

  static int parse(int argc, char *argv[]) {
    std::time_t t = std::time(nullptr);
    std::tm tm = *std::localtime(&t);
    // format date and time
    std::ostringstream date_and_time_ss;
    date_and_time_ss << std::put_time(&tm, "%Y%m%d%H%M%S");
    std::string log_dir_default = "log_" + date_and_time_ss.str();
    std::string config_file;

    po::options_description desc("General options");
    desc.add_options()
        ("help", "Display help message")
        ("config", po::value<std::string>(&config_file)->default_value(""),
         "name of a file of a configuration.")
        ("save_log_file",
         po::value<int>(&save_log_file)->default_value(0),
         "save log file to log directory")
        ("print_to_screen",
         po::value<bool>(&print_to_screen)->default_value(false),
         "Print info to screen")
        ("dim_order_opt",
         po::value<std::string>(&dim_order_opt)->default_value(""),
         "The dimension orders optimization method(random, genetic, traversal)")
        ("store_whole_block",
         po::value<bool>(&store_whole_block)->default_value(true),
         "Compute whole block size in mem footprint");

    po::options_description io("File IO options");
    io.add_options()
        ("log_directory",
         po::value<std::string>(&log_directory)->default_value(log_dir_default),
         "the name of log directory");

    po::options_description hw_constraints("Hardware constraints options");
    hw_constraints.add_options()
        ("mem_size", po::value<long>(&mem_size)->default_value(LONG_MAX), "memory size constraint")
        ("enable_compute_utilization_constraint",
         po::value<bool>(&enable_compute_utilization_constraint)->default_value(false),
         "enable compute utilization constraint")
        ("compute_power", po::value<long>(&compute_power)->default_value(1024), "compute power");

    po::options_description app_config("Application configuration options");
    app_config.add_options()
        ("seq_length", po::value<long>(&seq_length)->default_value(0), "sequence length")
        ("hid_size", po::value<long>(&hid_size)->default_value(0), "hidden dimension size")
        ("head_num", po::value<long>(&head_num)->default_value(0), "head number")
        ("head_blocksize", po::value<long>(&head_blocksize)->default_value(0), "head block size")
        ("batch_size", po::value<long>(&batch_size)->default_value(0), "batch size")
        ("batch_blocksize", po::value<long>(&batch_blocksize)->default_value(0),
         "batch block size");

    po::options_description all_options("Allows options");
    all_options.add(desc);
    all_options.add(io);
    all_options.add(hw_constraints);
    all_options.add(app_config);

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, all_options), vm);
    po::notify(vm);

    std::ifstream ifs(config_file.c_str());
    if (!ifs) {
      std::cout << "Warning: can not open config file: " << config_file << "\n";
    } else {
      store(parse_config_file(ifs, all_options), vm);
      notify(vm);
    }

    if (vm.count("help")) {
      std::cout << "Usage: DAT [options]\n";
      std::cout << all_options;
      exit(0);
    }

    if (!fs::exists(log_directory) && save_log_file)
      fs::create_directory(log_directory);

    return 0;
  }
};

std::string Options::log_directory;
bool Options::print_to_screen;
int Options::save_log_file;
bool Options::store_whole_block = true;
long Options::mem_size;
long Options::seq_length;
long Options::hid_size;
long Options::head_num;
long Options::head_blocksize;
long Options::batch_size;
long Options::batch_blocksize;
std::string Options::dim_order_opt;
long Options::compute_power;
bool Options::enable_compute_utilization_constraint;

}

#endif //MMCHAIN_ANALYSIS_SRC_OPTIONS_H
