package main

import (
	"deepseek-ocr-client/internal/api"
	"deepseek-ocr-client/internal/output"
	"deepseek-ocr-client/internal/scanner"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"

	"github.com/k0kubun/go-ansi"
	"github.com/schollz/progressbar/v3"
	"gopkg.in/yaml.v3"
)

// Job represents a PDF processing job
type Job struct {
	PDFPath      string
	RelativePath string // Path relative to scan root
	OutputPath   string
}

// Result represents the result of processing a PDF
type Result struct {
	Job          Job
	Success      bool
	Markdown     string
	Pages        int
	Error        error
	ProcessingMs int
}

// Stats tracks processing statistics
type Stats struct {
	Total       int64
	Processed   int64
	Success     int64
	Failed      int64
	TotalPages  int64
	TotalTimeMs int64
	StartTime   time.Time
}

// Config represents the application configuration
type Config struct {
	API struct {
		URL     string `yaml:"url"`
		Timeout int    `yaml:"timeout"`
	} `yaml:"api"`
	Processing struct {
		Workers       int `yaml:"workers"`
		RetryAttempts int `yaml:"retry_attempts"`
		RetryDelay    int `yaml:"retry_delay"`
	} `yaml:"processing"`
	Input struct {
		ScanDirectory string   `yaml:"scan_directory"`
		Recursive     bool     `yaml:"recursive"`
		Extensions    []string `yaml:"extensions"`
	} `yaml:"input"`
	Output struct {
		Directory          string `yaml:"directory"`
		PreserveStructure  bool   `yaml:"preserve_structure"`
		PageSeparator      string `yaml:"page_separator"`
		OverwriteExisting  bool   `yaml:"overwrite_existing"`
	} `yaml:"output"`
}

var (
	stats  Stats
	config Config
)

func main() {
	// Parse command-line flags
	apiURL := flag.String("api", "", "DeepSeek-OCR API URL (overrides config)")
	workers := flag.Int("workers", 0, "Number of worker goroutines (required)")
	scanDir := flag.String("scan", "", "Directory to scan for PDFs (required)")
	outputDir := flag.String("output", "", "Output directory for markdown files (overrides config)")
	recursive := flag.Bool("recursive", true, "Recursively scan directories")
	configFile := flag.String("config", "config.yaml", "Configuration file path")
	verbose := flag.Bool("verbose", false, "Enable verbose logging")
	overwrite := flag.Bool("overwrite", false, "Overwrite existing markdown files")
	flag.Parse()

	// Validate required flags
	if *workers <= 0 {
		fmt.Fprintf(os.Stderr, "Error: -workers flag is required and must be positive\n")
		flag.Usage()
		os.Exit(1)
	}

	if *scanDir == "" {
		fmt.Fprintf(os.Stderr, "Error: -scan flag is required\n")
		flag.Usage()
		os.Exit(1)
	}

	// Load configuration
	if err := loadConfig(*configFile); err != nil {
		log.Printf("Warning: Could not load config file: %v", err)
		// Set defaults
		config.API.URL = "http://localhost:8000"
		config.API.Timeout = 300
		config.Processing.RetryAttempts = 3
		config.Processing.RetryDelay = 5
		config.Output.Directory = "./output"
		config.Output.PreserveStructure = true
		config.Output.PageSeparator = "<!-- Page %d -->"
	}

	// Override config with command-line flags
	if *apiURL != "" {
		config.API.URL = *apiURL
	}
	config.Processing.Workers = *workers
	config.Input.ScanDirectory = *scanDir
	config.Input.Recursive = *recursive
	config.Output.OverwriteExisting = *overwrite
	if *outputDir != "" {
		config.Output.Directory = *outputDir
	}

	// Set logging level
	if !*verbose {
		log.SetOutput(&quietLogger{})
	}

	// Initialize statistics
	stats.StartTime = time.Now()

	// Print startup information
	fmt.Printf("DeepSeek-OCR Client v1.0.0\n")
	fmt.Printf("=============================\n")
	fmt.Printf("API URL:    %s\n", config.API.URL)
	fmt.Printf("Workers:    %d\n", config.Processing.Workers)
	fmt.Printf("Scan Dir:   %s\n", config.Input.ScanDirectory)
	fmt.Printf("Output Dir: %s\n", config.Output.Directory)
	fmt.Printf("Recursive:  %v\n", config.Input.Recursive)
	fmt.Printf("Overwrite:  %v\n", config.Output.OverwriteExisting)
	fmt.Printf("=============================\n\n")

	// Check API health before starting
	client := api.NewClient(config.API.URL, config.API.Timeout)
	if err := client.HealthCheck(); err != nil {
		fmt.Fprintf(os.Stderr, "Warning: API health check failed: %v\n", err)
		fmt.Fprintf(os.Stderr, "Continuing anyway...\n\n")
	} else {
		fmt.Printf("✓ API is healthy\n\n")
	}

	// Create channels
	jobs := make(chan Job, 100)
	results := make(chan Result, 100)

	// Create WaitGroups
	var workersWG sync.WaitGroup
	var collectorWG sync.WaitGroup

	// Start scanner in a goroutine
	go func() {
		defer close(jobs)
		s := scanner.NewScanner(config.Input.ScanDirectory, config.Output.Directory)
		if err := s.ScanForPDFs(config.Input.Recursive, config.Output.OverwriteExisting, jobs, &stats); err != nil {
			log.Fatalf("Scanner error: %v", err)
		}
		log.Println("Scanner: Finished discovering PDFs")
	}()

	// Wait a moment for scanner to discover some files
	time.Sleep(100 * time.Millisecond)
	total := atomic.LoadInt64(&stats.Total)

	if total == 0 {
		// Wait a bit more
		time.Sleep(500 * time.Millisecond)
		total = atomic.LoadInt64(&stats.Total)
		if total == 0 {
			fmt.Println("No PDF files found to process")
			return
		}
	}

	// Create progress bar
	bar := progressbar.NewOptions(int(total),
		progressbar.OptionSetWriter(ansi.NewAnsiStdout()),
		progressbar.OptionEnableColorCodes(true),
		progressbar.OptionSetWidth(40),
		progressbar.OptionShowCount(),
		progressbar.OptionSetDescription("Processing PDFs"),
		progressbar.OptionSetTheme(progressbar.Theme{
			Saucer:        "[green]█[reset]",
			SaucerHead:    "[green]█[reset]",
			SaucerPadding: "░",
			BarStart:      "│",
			BarEnd:        "│",
		}),
		progressbar.OptionOnCompletion(func() {
			fmt.Println()
		}),
	)

	// Start workers
	for i := 0; i < config.Processing.Workers; i++ {
		workersWG.Add(1)
		go worker(i+1, client, jobs, results, &workersWG)
	}

	// Start result collector
	collectorWG.Add(1)
	go collector(results, bar, &collectorWG)

	// Wait for workers to finish
	workersWG.Wait()
	close(results)
	log.Println("Workers: All finished")

	// Wait for collector to finish
	collectorWG.Wait()

	// Print final statistics
	printStats(&stats)
}

func worker(id int, client *api.Client, jobs <-chan Job, results chan<- Result, wg *sync.WaitGroup) {
	defer wg.Done()

	log.Printf("Worker %d: Started", id)

	for job := range jobs {
		log.Printf("Worker %d: Processing %s", id, job.RelativePath)

		result := processJob(client, job)
		results <- result

		if result.Success {
			log.Printf("Worker %d: Successfully processed %s (%d pages in %dms)",
				id, job.RelativePath, result.Pages, result.ProcessingMs)
		} else {
			log.Printf("Worker %d: Failed to process %s: %v",
				id, job.RelativePath, result.Error)
		}
	}

	log.Printf("Worker %d: Finished", id)
}

func processJob(client *api.Client, job Job) Result {
	startTime := time.Now()

	// Process PDF with retry logic
	var markdown string
	var pages int
	var err error

	for attempt := 1; attempt <= config.Processing.RetryAttempts; attempt++ {
		markdown, pages, err = client.ProcessPDF(job.PDFPath)
		if err == nil {
			break
		}

		if attempt < config.Processing.RetryAttempts {
			log.Printf("Attempt %d failed for %s: %v. Retrying in %d seconds...",
				attempt, job.RelativePath, err, config.Processing.RetryDelay)
			time.Sleep(time.Duration(config.Processing.RetryDelay) * time.Second)
		}
	}

	processingMs := int(time.Since(startTime).Milliseconds())

	if err != nil {
		atomic.AddInt64(&stats.Failed, 1)
		return Result{
			Job:          job,
			Success:      false,
			Error:        err,
			ProcessingMs: processingMs,
		}
	}

	atomic.AddInt64(&stats.Success, 1)
	atomic.AddInt64(&stats.TotalPages, int64(pages))
	atomic.AddInt64(&stats.TotalTimeMs, int64(processingMs))

	return Result{
		Job:          job,
		Success:      true,
		Markdown:     markdown,
		Pages:        pages,
		ProcessingMs: processingMs,
	}
}

func collector(results <-chan Result, bar *progressbar.ProgressBar, wg *sync.WaitGroup) {
	defer wg.Done()

	writer := output.NewWriter(config.Output.Directory)

	for result := range results {
		atomic.AddInt64(&stats.Processed, 1)

		if result.Success {
			// Write markdown file
			err := writer.WriteMarkdownFile(result.Job.OutputPath, result.Markdown)
			if err != nil {
				log.Printf("Error writing %s: %v", result.Job.OutputPath, err)
				// Convert success to failure
				atomic.AddInt64(&stats.Success, -1)
				atomic.AddInt64(&stats.Failed, 1)
			}
		}

		bar.Add(1)
	}
}

func loadConfig(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}

	return yaml.Unmarshal(data, &config)
}

func printStats(stats *Stats) {
	duration := time.Since(stats.StartTime)

	fmt.Println("\n=============================")
	fmt.Println("Processing Complete")
	fmt.Println("=============================")
	fmt.Printf("Total PDFs:        %d\n", stats.Total)
	fmt.Printf("Successful:        %d\n", stats.Success)
	fmt.Printf("Failed:            %d\n", stats.Failed)
	fmt.Printf("Total Pages:       %d\n", stats.TotalPages)

	if stats.Success > 0 {
		avgPages := float64(stats.TotalPages) / float64(stats.Success)
		avgTime := float64(stats.TotalTimeMs) / float64(stats.Success)
		fmt.Printf("Avg Pages/PDF:     %.1f\n", avgPages)
		fmt.Printf("Avg Time/PDF:      %.1f seconds\n", avgTime/1000)
		fmt.Printf("Throughput:        %.1f pages/min\n",
			float64(stats.TotalPages)/(duration.Minutes()))
	}

	fmt.Printf("Total Time:        %v\n", duration.Round(time.Second))
	fmt.Println("=============================")

	// Exit with error code if any failures
	if stats.Failed > 0 {
		os.Exit(1)
	}
}

// quietLogger suppresses log output when verbose is false
type quietLogger struct{}

func (q *quietLogger) Write(p []byte) (n int, err error) {
	return len(p), nil
}