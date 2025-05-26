// Main JavaScript for Organic Fruit Classification

document.addEventListener('DOMContentLoaded', function() {
    // Initialize all components
    initFileUpload();
    initAnimations();
    initSmoothScroll();
    initNavbar();
});

// File Upload Functionality
function initFileUpload() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const previewArea = document.getElementById('previewArea');
    const previewImage = document.getElementById('previewImage');
    const fileName = document.getElementById('fileName');
    const classifyBtn = document.getElementById('classifyBtn');
    const removeBtn = document.getElementById('removeImage');
    const form = document.getElementById('classifyForm');

    if (!uploadArea || !fileInput) return;

    // Drag and drop functionality
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });

    // File input change
    fileInput.addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    // Remove image
    if (removeBtn) {
        removeBtn.addEventListener('click', function() {
            resetFileUpload();
        });
    }

    // Form submission
    if (form) {
        form.addEventListener('submit', function(e) {
            if (!fileInput.files.length) {
                e.preventDefault();
                showAlert('Please select an image file first.', 'error');
                return;
            }
            
            // Show loading state
            showLoadingState();
        });
    }

    function handleFileSelect(file) {
        // Validate file type
        const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp', 'image/webp'];
        if (!allowedTypes.includes(file.type)) {
            showAlert('Please select a valid image file (JPEG, PNG, GIF, BMP, WebP).', 'error');
            return;
        }

        // Validate file size (16MB max)
        const maxSize = 16 * 1024 * 1024;
        if (file.size > maxSize) {
            showAlert('File size too large. Please select an image smaller than 16MB.', 'error');
            return;
        }

        // Create FileList for the input
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        fileInput.files = dataTransfer.files;

        // Show preview
        showImagePreview(file);
    }

    function showImagePreview(file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            previewImage.src = e.target.result;
            fileName.textContent = file.name;
            
            // Hide upload content, show preview
            uploadArea.querySelector('.upload-content').style.display = 'none';
            previewArea.style.display = 'block';
            
            // Enable classify button
            if (classifyBtn) {
                classifyBtn.disabled = false;
                classifyBtn.classList.add('btn-pulse');
            }
        };
        reader.readAsDataURL(file);
    }

    function resetFileUpload() {
        fileInput.value = '';
        uploadArea.querySelector('.upload-content').style.display = 'block';
        previewArea.style.display = 'none';
        
        if (classifyBtn) {
            classifyBtn.disabled = true;
            classifyBtn.classList.remove('btn-pulse');
        }
    }

    function showLoadingState() {
        if (classifyBtn) {
            classifyBtn.disabled = true;
            classifyBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
        }
    }
}

// Animation functionality
function initAnimations() {
    // Intersection Observer for scroll animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-in');
            }
        });
    }, observerOptions);

    // Observe elements for animation
    const animateElements = document.querySelectorAll('.feature-card, .fruit-item, .related-fruit-card, .benefit-card');
    animateElements.forEach(el => {
        observer.observe(el);
    });

    // Add CSS for animations
    if (!document.getElementById('dynamic-animations')) {
        const style = document.createElement('style');
        style.id = 'dynamic-animations';
        style.textContent = `
            .feature-card, .fruit-item, .related-fruit-card, .benefit-card {
                opacity: 0;
                transform: translateY(30px);
                transition: all 0.6s ease;
            }
            .animate-in {
                opacity: 1 !important;
                transform: translateY(0) !important;
            }
            .btn-pulse {
                animation: pulse 2s infinite;
            }
            @keyframes pulse {
                0% { box-shadow: 0 0 0 0 rgba(139, 195, 74, 0.7); }
                70% { box-shadow: 0 0 0 10px rgba(139, 195, 74, 0); }
                100% { box-shadow: 0 0 0 0 rgba(139, 195, 74, 0); }
            }
        `;
        document.head.appendChild(style);
    }
}

// Smooth scrolling for anchor links
function initSmoothScroll() {
    const links = document.querySelectorAll('a[href^="#"]');
    
    links.forEach(link => {
        link.addEventListener('click', function(e) {
            const href = this.getAttribute('href');
            if (href === '#') return;
            
            const target = document.querySelector(href);
            if (target) {
                e.preventDefault();
                
                const offsetTop = target.offsetTop - 80; // Account for fixed navbar
                
                window.scrollTo({
                    top: offsetTop,
                    behavior: 'smooth'
                });
            }
        });
    });
}

// Navbar behavior
function initNavbar() {
    const navbar = document.querySelector('.navbar');
    if (!navbar) return;

    let lastScrollY = window.scrollY;

    window.addEventListener('scroll', function() {
        const currentScrollY = window.scrollY;
        
        // Add/remove shadow based on scroll position
        if (currentScrollY > 50) {
            navbar.style.boxShadow = '0 2px 20px rgba(0, 0, 0, 0.1)';
        } else {
            navbar.style.boxShadow = 'none';
        }

        lastScrollY = currentScrollY;
    });

    // Close mobile menu when clicking outside
    document.addEventListener('click', function(e) {
        const navbarCollapse = document.querySelector('.navbar-collapse');
        const navbarToggler = document.querySelector('.navbar-toggler');
        
        if (navbarCollapse && navbarCollapse.classList.contains('show')) {
            if (!navbarCollapse.contains(e.target) && !navbarToggler.contains(e.target)) {
                const bsCollapse = new bootstrap.Collapse(navbarCollapse);
                bsCollapse.hide();
            }
        }
    });
}

// Utility function to show alerts
function showAlert(message, type = 'info') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type === 'error' ? 'danger' : 'success'} alert-dismissible fade show`;
    alertDiv.style.position = 'fixed';
    alertDiv.style.top = '100px';
    alertDiv.style.right = '20px';
    alertDiv.style.zIndex = '1050';
    alertDiv.style.maxWidth = '400px';
    
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(alertDiv);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.remove();
        }
    }, 5000);
}

// Parallax effect for hero background
function initParallax() {
    const heroBackground = document.querySelector('.hero-background');
    if (!heroBackground) return;

    window.addEventListener('scroll', function() {
        const scrolled = window.pageYOffset;
        const parallax = scrolled * 0.5;
        
        heroBackground.style.transform = `translateY(${parallax}px)`;
    });
}

// Initialize parallax effect
initParallax();

// Image lazy loading for better performance
function initLazyLoading() {
    const images = document.querySelectorAll('img[data-src]');
    
    const imageObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const img = entry.target;
                img.src = img.dataset.src;
                img.classList.remove('lazy');
                imageObserver.unobserve(img);
            }
        });
    });

    images.forEach(img => imageObserver.observe(img));
}

// Enhanced form validation
function validateForm(formElement) {
    const inputs = formElement.querySelectorAll('input[required], select[required], textarea[required]');
    let isValid = true;

    inputs.forEach(input => {
        if (!input.value.trim()) {
            showFieldError(input, 'This field is required');
            isValid = false;
        } else {
            clearFieldError(input);
        }
    });

    return isValid;
}

function showFieldError(input, message) {
    clearFieldError(input);
    
    input.classList.add('is-invalid');
    
    const errorDiv = document.createElement('div');
    errorDiv.className = 'invalid-feedback';
    errorDiv.textContent = message;
    
    input.parentNode.appendChild(errorDiv);
}

function clearFieldError(input) {
    input.classList.remove('is-invalid');
    
    const errorDiv = input.parentNode.querySelector('.invalid-feedback');
    if (errorDiv) {
        errorDiv.remove();
    }
}

// Page transition effects
function initPageTransitions() {
    // Add fade-in effect to page content
    document.body.style.opacity = '0';
    
    window.addEventListener('load', function() {
        document.body.style.transition = 'opacity 0.5s ease-in-out';
        document.body.style.opacity = '1';
    });
}

// Initialize page transitions
initPageTransitions();

// Export functions for use in other scripts
window.FruitClassifier = {
    showAlert,
    validateForm,
    initAnimations,
    initSmoothScroll
};
