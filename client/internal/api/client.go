package api

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// Client handles API communication with the DeepSeek-OCR server
type Client struct {
	baseURL    string
	httpClient *http.Client
}

// PDFResponse represents the API response for PDF processing
type PDFResponse struct {
	Success               bool         `json:"success"`
	Filename              string       `json:"filename"`
	TotalPages            int          `json:"total_pages"`
	Pages                 []PageResult `json:"pages"`
	TotalProcessingTimeMs int          `json:"total_processing_time_ms"`
}

// PageResult represents the result for a single page
type PageResult struct {
	PageNumber      int    `json:"page_number"`
	Markdown        string `json:"markdown"`
	ProcessingTimeMs int   `json:"processing_time_ms"`
}

// HealthResponse represents the health check response
type HealthResponse struct {
	Status         string  `json:"status"`
	ModelLoaded    bool    `json:"model_loaded"`
	CudaAvailable  bool    `json:"cuda_available"`
	GpuMemoryFree  string  `json:"gpu_memory_free"`
	GpuMemoryTotal string  `json:"gpu_memory_total,omitempty"`
	ModelPath      string  `json:"model_path,omitempty"`
	UptimeSeconds  float64 `json:"uptime_seconds,omitempty"`
}

// ErrorResponse represents an error from the API
type ErrorResponse struct {
	Error  string `json:"error"`
	Detail string `json:"detail,omitempty"`
}

// NewClient creates a new API client
func NewClient(baseURL string, timeoutSeconds int) *Client {
	// Ensure baseURL doesn't have trailing slash
	baseURL = strings.TrimRight(baseURL, "/")

	return &Client{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: time.Duration(timeoutSeconds) * time.Second,
		},
	}
}

// HealthCheck checks if the API is healthy
func (c *Client) HealthCheck() error {
	url := fmt.Sprintf("%s/health", c.baseURL)

	resp, err := c.httpClient.Get(url)
	if err != nil {
		return fmt.Errorf("health check request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("health check failed with status %d: %s", resp.StatusCode, string(body))
	}

	var health HealthResponse
	if err := json.NewDecoder(resp.Body).Decode(&health); err != nil {
		return fmt.Errorf("failed to decode health response: %w", err)
	}

	if health.Status != "healthy" {
		return fmt.Errorf("API is not healthy: %s", health.Status)
	}

	if !health.ModelLoaded {
		return fmt.Errorf("model is not loaded")
	}

	return nil
}

// ProcessPDF sends a PDF file to the API for processing
func (c *Client) ProcessPDF(pdfPath string) (markdown string, pages int, err error) {
	// Open the PDF file
	file, err := os.Open(pdfPath)
	if err != nil {
		return "", 0, fmt.Errorf("failed to open PDF file: %w", err)
	}
	defer file.Close()

	// Get file info
	fileInfo, err := file.Stat()
	if err != nil {
		return "", 0, fmt.Errorf("failed to get file info: %w", err)
	}

	// Create multipart form
	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)

	// Add the file part
	part, err := writer.CreateFormFile("file", filepath.Base(pdfPath))
	if err != nil {
		return "", 0, fmt.Errorf("failed to create form file: %w", err)
	}

	// Copy file content
	_, err = io.Copy(part, file)
	if err != nil {
		return "", 0, fmt.Errorf("failed to copy file content: %w", err)
	}

	// Add resolution field (optional)
	writer.WriteField("resolution", "base")

	// Close the writer
	err = writer.Close()
	if err != nil {
		return "", 0, fmt.Errorf("failed to close multipart writer: %w", err)
	}

	// Create the request
	url := fmt.Sprintf("%s/api/v1/ocr/pdf", c.baseURL)
	req, err := http.NewRequest("POST", url, body)
	if err != nil {
		return "", 0, fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers
	req.Header.Set("Content-Type", writer.FormDataContentType())

	// Log request details
	// fmt.Printf("Sending %s (%.2f MB) to %s\n", filepath.Base(pdfPath), float64(fileInfo.Size())/(1024*1024), url)

	// Send the request
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return "", 0, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	// Read response body
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", 0, fmt.Errorf("failed to read response: %w", err)
	}

	// Check status code
	if resp.StatusCode != http.StatusOK {
		var errorResp ErrorResponse
		if err := json.Unmarshal(respBody, &errorResp); err == nil {
			return "", 0, fmt.Errorf("API error: %s - %s", errorResp.Error, errorResp.Detail)
		}
		return "", 0, fmt.Errorf("API returned status %d: %s", resp.StatusCode, string(respBody))
	}

	// Parse response
	var pdfResp PDFResponse
	if err := json.Unmarshal(respBody, &pdfResp); err != nil {
		return "", 0, fmt.Errorf("failed to decode response: %w", err)
	}

	if !pdfResp.Success {
		return "", 0, fmt.Errorf("processing failed")
	}

	// Combine all pages into single markdown
	var combined strings.Builder
	for _, page := range pdfResp.Pages {
		// Add page separator
		combined.WriteString(fmt.Sprintf("<!-- Page %d -->\n\n", page.PageNumber))
		// Add page content
		combined.WriteString(page.Markdown)
		// Add spacing between pages
		combined.WriteString("\n\n---\n\n")
	}

	return combined.String(), pdfResp.TotalPages, nil
}

// ProcessPDFWithRetry processes a PDF with retry logic
func (c *Client) ProcessPDFWithRetry(pdfPath string, maxRetries int, retryDelay time.Duration) (string, int, error) {
	var lastErr error

	for i := 0; i < maxRetries; i++ {
		markdown, pages, err := c.ProcessPDF(pdfPath)
		if err == nil {
			return markdown, pages, nil
		}

		lastErr = err

		// Check if error is retryable
		if !isRetryableError(err) {
			return "", 0, err
		}

		// Wait before retry (except on last attempt)
		if i < maxRetries-1 {
			time.Sleep(retryDelay)
		}
	}

	return "", 0, fmt.Errorf("failed after %d attempts: %w", maxRetries, lastErr)
}

// GetStatus gets the server status
func (c *Client) GetStatus() (map[string]interface{}, error) {
	url := fmt.Sprintf("%s/api/v1/status", c.baseURL)

	resp, err := c.httpClient.Get(url)
	if err != nil {
		return nil, fmt.Errorf("status request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("status request failed with status %d: %s", resp.StatusCode, string(body))
	}

	var status map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&status); err != nil {
		return nil, fmt.Errorf("failed to decode status response: %w", err)
	}

	return status, nil
}

// ProcessPDFFromURL processes a PDF from a URL
func (c *Client) ProcessPDFFromURL(pdfURL string, resolution string) (string, int, error) {
	// Create request body
	reqBody := map[string]string{
		"url":        pdfURL,
		"resolution": resolution,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return "", 0, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create the request
	url := fmt.Sprintf("%s/api/v1/ocr/pdf-url", c.baseURL)
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return "", 0, fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers
	req.Header.Set("Content-Type", "application/json")

	// Send the request
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return "", 0, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	// Read response body
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", 0, fmt.Errorf("failed to read response: %w", err)
	}

	// Check status code
	if resp.StatusCode != http.StatusOK {
		var errorResp ErrorResponse
		if err := json.Unmarshal(respBody, &errorResp); err == nil {
			return "", 0, fmt.Errorf("API error: %s - %s", errorResp.Error, errorResp.Detail)
		}
		return "", 0, fmt.Errorf("API returned status %d: %s", resp.StatusCode, string(respBody))
	}

	// Parse response
	var pdfResp PDFResponse
	if err := json.Unmarshal(respBody, &pdfResp); err != nil {
		return "", 0, fmt.Errorf("failed to decode response: %w", err)
	}

	if !pdfResp.Success {
		return "", 0, fmt.Errorf("processing failed")
	}

	// Combine all pages
	var combined strings.Builder
	for _, page := range pdfResp.Pages {
		combined.WriteString(fmt.Sprintf("<!-- Page %d -->\n\n", page.PageNumber))
		combined.WriteString(page.Markdown)
		combined.WriteString("\n\n---\n\n")
	}

	return combined.String(), pdfResp.TotalPages, nil
}

// isRetryableError determines if an error should trigger a retry
func isRetryableError(err error) bool {
	errStr := err.Error()

	// Network errors
	if strings.Contains(errStr, "connection refused") ||
		strings.Contains(errStr, "connection reset") ||
		strings.Contains(errStr, "timeout") ||
		strings.Contains(errStr, "temporary failure") {
		return true
	}

	// Server errors (5xx)
	if strings.Contains(errStr, "status 500") ||
		strings.Contains(errStr, "status 502") ||
		strings.Contains(errStr, "status 503") ||
		strings.Contains(errStr, "status 504") {
		return true
	}

	return false
}