#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <curlpp/Easy.hpp>
#include <curlpp/Options.hpp>
#include <curlpp/cURLpp.hpp>

/**
 * @brief Download HTML from a single URL and save it to a std::stringstream.
 *
 * @param url
 * @return std::stringstream
 */
std::stringstream download(std::string url) {
  std::stringstream ss;

  try {
    // curlpp::Cleanup myCleanup;
    ss << curlpp::options::Url(url);

    return ss;
  }

  catch (curlpp::RuntimeError &e) {
    std::cout << e.what() << std::endl;
  }

  catch (curlpp::LogicError &e) {
    std::cout << e.what() << std::endl;
  }
  return ss;
}

using data_dict = std::unordered_map<std::string, std::stringstream>;

/**
 * @brief Download multiple URLs and save into a dictionary where the URL is the
 * key.
 *
 * @param urls
 * @return data_dict
 */
data_dict download_multiple_serial(std::vector<std::string> &urls) {
  data_dict dd;

  for (std::string const &url : urls) {
    dd[url] = download(url);
  }

  return dd;
}

data_dict download_multiple_gather(std::vector<std::string> &urls) {
  data_dict final_dd;

#pragma omp parallel
  {
    data_dict dd;
    for (std::string const &url : urls) {
      dd[url] = download(url);
    }

#pragma omp critical
    { final_dd.merge(dd); }
  }

  return final_dd;
}

/*
 * 1) Serial
 * 2) Parallel with gather
 * (https://stackoverflow.com/questions/18669296/c-openmp-parallel-for-loop-alternatives-to-stdvector)
 * 3) Parallel with threadsafe container see slide 10
 * (https://openmpcon.org/wp-content/uploads/2018_Tutorial3_Martorell_Teruel_Klemm.pdf)
 */

/**
 * This example is made to show you how you can use the Options.
 */

typedef enum {
  SERIAL = 1,
  PARALLEL_GATHER = 2,
  PARALLEL_CONTAINER = 3,
} DOWNLOADER_TYPE;

int main(int argc, char **argv) {

  DOWNLOADER_TYPE downloader;

  if (argc != 2) {
    std::cout << "Defaulting to serial downloader" << std::endl;
    downloader = SERIAL;
  } else {
    downloader = static_cast<DOWNLOADER_TYPE>(std::stoi(argv[1]));
  }

  // From the libcurl docs:
  std::vector<std::string> urls = {
      "https://www.microsoft.com",
      "https://opensource.org",
      "https://www.google.com",
      "https://www.yahoo.com",
      "https://www.ibm.com",
      "https://www.mysql.com",
      "https://www.oracle.com",
      "https://www.ripe.net",
      "https://www.iana.org",
      "https://www.amazon.com",
      "https://www.netcraft.com",
      "https://www.heise.de",
      "https://www.chip.de",
      "https://www.ca.com",
      "https://www.cnet.com",
      "https://www.mozilla.org",
      "https://www.cnn.com",
      "https://www.wikipedia.org",
      "https://www.dell.com",
      "https://www.hp.com",
      "https://www.cert.org",
      "https://www.mit.edu",
      "https://www.nist.gov",
      "https://www.ebay.com",
      "https://www.playstation.com",
      "https://www.uefa.com",
      "https://www.ieee.org",
      "https://www.apple.com",
      "https://www.symantec.com",
      "https://www.zdnet.com",
      "https://www.fujitsu.com/global/",
      "https://www.supermicro.com",
      "https://www.hotmail.com",
      "https://www.ietf.org",
      "https://www.bbc.co.uk",
      "https://news.google.com",
      "https://www.foxnews.com",
      "https://www.msn.com",
      "https://www.wired.com",
      "https://www.sky.com",
      "https://www.usatoday.com",
      "https://www.cbs.com",
      "https://www.nbc.com/",
      "https://slashdot.org",
      "https://www.informationweek.com",
      "https://apache.org",
      "https://www.un.org",
  };

  data_dict dd;
  switch (downloader) {
  case SERIAL:
    dd = download_multiple_serial(urls);
    break;
  case PARALLEL_GATHER:
    dd = download_multiple_gather(urls);
    break;
  default:
    std::cerr << "[ERROR]: Bad user argument for download method." << std::endl;
    return -1;
  }
  return 0;
}