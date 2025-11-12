package scanner

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
)

// Job represents a PDF processing job
type Job struct {
	PDFPath      string
	RelativePath string
	OutputPath   string
}

// Stats interface for updating statistics
type Stats interface {
	AddTotal(delta int64)
}

// Scanner handles PDF file discovery
type Scanner struct {
	scanDir   string
	outputDir string
}

// NewScanner creates a new scanner instance
func NewScanner(scanDir, outputDir string) *Scanner {
	return &Scanner{
		scanDir:   scanDir,
		outputDir: outputDir,
	}
}

// ScanForPDFs scans for PDF files and sends them to the jobs channel
func (s *Scanner) ScanForPDFs(recursive, overwrite bool, jobs chan<- Job, stats interface{}) error {
	// Validate scan directory exists
	info, err := os.Stat(s.scanDir)
	if err != nil {
		return fmt.Errorf("scan directory error: %w", err)
	}
	if !info.IsDir() {
		return fmt.Errorf("scan path is not a directory: %s", s.scanDir)
	}

	// Create output directory if it doesn't exist
	if err := os.MkdirAll(s.outputDir, 0755); err != nil {
		return fmt.Errorf("failed to create output directory: %w", err)
	}

	log.Printf("Scanner: Starting scan of %s", s.scanDir)

	// Get type assertion for stats
	var statsCounter *int64
	if s, ok := stats.(interface{ Total int64 }); ok {
		statsCounter = &s.Total
	}

	// Walk the directory tree
	err = filepath.Walk(s.scanDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			log.Printf("Scanner: Error accessing %s: %v", path, err)
			return nil // Continue walking
		}

		// Skip directories if not recursive
		if info.IsDir() && !recursive && path != s.scanDir {
			return filepath.SkipDir
		}

		// Check if it's a PDF file
		if !info.IsDir() && isPDFFile(path) {
			// Calculate relative path for output structure
			relPath, err := filepath.Rel(s.scanDir, path)
			if err != nil {
				log.Printf("Scanner: Error calculating relative path for %s: %v", path, err)
				return nil
			}

			// Construct output path (replace .pdf with .md)
			outputPath := s.constructOutputPath(relPath)

			// Check if already processed (unless overwrite is enabled)
			if !overwrite {
				if _, err := os.Stat(outputPath); err == nil {
					log.Printf("Scanner: Skipping %s (already processed)", relPath)
					return nil
				}
			}

			// Add to job queue
			job := Job{
				PDFPath:      path,
				RelativePath: relPath,
				OutputPath:   outputPath,
			}

			// Send job to channel
			select {
			case jobs <- job:
				if statsCounter != nil {
					atomic.AddInt64(statsCounter, 1)
				}
				log.Printf("Scanner: Found %s", relPath)
			default:
				// Channel is full, wait
				jobs <- job
				if statsCounter != nil {
					atomic.AddInt64(statsCounter, 1)
				}
				log.Printf("Scanner: Found %s (queue was full, waited)", relPath)
			}
		}

		return nil
	})

	if err != nil {
		return fmt.Errorf("error walking directory: %w", err)
	}

	log.Printf("Scanner: Scan complete")
	return nil
}

// ScanForPDFsBatch scans and returns all PDFs at once (alternative method)
func (s *Scanner) ScanForPDFsBatch(recursive, overwrite bool) ([]Job, error) {
	var jobs []Job

	err := filepath.Walk(s.scanDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil // Skip files with errors
		}

		// Skip directories if not recursive
		if info.IsDir() && !recursive && path != s.scanDir {
			return filepath.SkipDir
		}

		// Check if it's a PDF file
		if !info.IsDir() && isPDFFile(path) {
			// Calculate relative path
			relPath, err := filepath.Rel(s.scanDir, path)
			if err != nil {
				return nil
			}

			// Construct output path
			outputPath := s.constructOutputPath(relPath)

			// Check if already processed
			if !overwrite {
				if _, err := os.Stat(outputPath); err == nil {
					return nil // Skip
				}
			}

			jobs = append(jobs, Job{
				PDFPath:      path,
				RelativePath: relPath,
				OutputPath:   outputPath,
			})
		}

		return nil
	})

	return jobs, err
}

// constructOutputPath creates the output path for a markdown file
func (s *Scanner) constructOutputPath(relativePath string) string {
	// Remove .pdf extension and add .md
	mdPath := strings.TrimSuffix(relativePath, filepath.Ext(relativePath)) + ".md"

	// Construct full output path
	outputPath := filepath.Join(s.outputDir, mdPath)

	// Ensure output directory exists
	outputDir := filepath.Dir(outputPath)
	os.MkdirAll(outputDir, 0755)

	return outputPath
}

// isPDFFile checks if a file is a PDF based on extension
func isPDFFile(path string) bool {
	ext := strings.ToLower(filepath.Ext(path))
	return ext == ".pdf"
}

// GetPDFInfo returns basic information about a PDF file
func GetPDFInfo(path string) (map[string]interface{}, error) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, err
	}

	return map[string]interface{}{
		"path":     path,
		"size":     info.Size(),
		"modified": info.ModTime(),
		"name":     info.Name(),
	}, nil
}

// EstimatePDFCount estimates the number of PDFs in a directory
func EstimatePDFCount(scanDir string, recursive bool) (int, error) {
	count := 0

	err := filepath.Walk(scanDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil // Skip errors
		}

		// Skip directories if not recursive
		if info.IsDir() && !recursive && path != scanDir {
			return filepath.SkipDir
		}

		// Count PDF files
		if !info.IsDir() && isPDFFile(path) {
			count++
		}

		return nil
	})

	return count, err
}

// ValidatePath checks if a path exists and is accessible
func ValidatePath(path string) error {
	info, err := os.Stat(path)
	if err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("path does not exist: %s", path)
		}
		return fmt.Errorf("cannot access path: %w", err)
	}

	if !info.IsDir() {
		return fmt.Errorf("path is not a directory: %s", path)
	}

	return nil
}