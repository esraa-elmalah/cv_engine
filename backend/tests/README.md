# CV Engine Test Suite

## 🏗️ Test Structure

The test suite follows a well-organized structure following Python testing best practices:

```
tests/
├── __init__.py                 # Makes tests a Python package
├── conftest.py                 # Pytest configuration and fixtures
├── unit/                       # Unit tests
│   ├── __init__.py
│   ├── test_cv_generator.py    # CV generator unit tests
│   └── test_image_validator.py # Image validation unit tests
├── integration/                # Integration tests
│   ├── __init__.py
│   └── test_cv_generation_integration.py
└── api/                        # API tests
    ├── __init__.py
    └── test_api_endpoints.py
```

## 🧪 Test Categories

### 1. **Unit Tests** (`tests/unit/`)
- **Purpose**: Test individual components in isolation
- **Scope**: Single functions, classes, or methods
- **Dependencies**: Mocked external dependencies
- **Speed**: Fast execution

**Examples:**
- CV template creation and validation
- Image validation logic
- Service initialization
- Configuration validation

### 2. **Integration Tests** (`tests/integration/`)
- **Purpose**: Test component interactions
- **Scope**: Multiple components working together
- **Dependencies**: Partially mocked external services
- **Speed**: Medium execution time

**Examples:**
- Complete CV generation flow
- Image validation integration
- Error handling scenarios
- Service orchestration

### 3. **API Tests** (`tests/api/`)
- **Purpose**: Test HTTP endpoints
- **Scope**: Full API request/response cycle
- **Dependencies**: Mocked service layer
- **Speed**: Fast execution

**Examples:**
- Endpoint responses
- Request validation
- Error handling
- Response formats

## 🚀 Running Tests

### Using the Test Runner Script

```bash
# Run all tests
python run_tests.py all

# Run unit tests only
python run_tests.py unit

# Run integration tests only
python run_tests.py integration

# Run API tests only
python run_tests.py api

# Run tests with coverage
python run_tests.py coverage

# Run quick tests (unit only, fast)
python run_tests.py quick

# Verbose output
python run_tests.py all --verbose

# Parallel execution
python run_tests.py all --parallel
```

### Using Pytest Directly

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/api/

# Run specific test files
pytest tests/unit/test_cv_generator.py
pytest tests/api/test_api_endpoints.py

# Run with coverage
pytest --cov=app --cov-report=html tests/

# Run with markers
pytest -m unit tests/
pytest -m integration tests/
pytest -m api tests/

# Run specific test functions
pytest tests/unit/test_cv_generator.py::TestCVGeneratorService::test_service_initialization
```

## 🔧 Test Configuration

### Pytest Configuration (`pytest.ini`)

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --tb=short
    --strict-markers
    --disable-warnings
    --color=yes
markers =
    unit: Unit tests
    integration: Integration tests
    api: API tests
    slow: Slow running tests
    external: Tests that require external services
asyncio_mode = auto
```

### Test Fixtures (`conftest.py`)

Common fixtures available to all tests:

- `cv_generator_config`: Test CV generator configuration
- `image_validation_config`: Test image validation configuration
- `sample_cv_template`: Sample CV template for testing
- `senior_cv_template`: Senior-level CV template
- `junior_cv_template`: Junior-level CV template
- `cv_generator_service`: CV generator service instance
- `image_validation_service`: Image validation service instance
- `mock_image_data`: Mock image data for testing
- `client`: FastAPI test client

## 📊 Test Coverage

### Coverage Reports

```bash
# Generate HTML coverage report
pytest --cov=app --cov-report=html tests/

# Generate terminal coverage report
pytest --cov=app --cov-report=term-missing tests/

# Generate both HTML and terminal reports
pytest --cov=app --cov-report=html --cov-report=term-missing tests/
```

### Coverage Targets

- **Unit Tests**: >90% coverage
- **Integration Tests**: >80% coverage
- **API Tests**: >95% coverage
- **Overall**: >85% coverage

## 🛠️ Writing Tests

### Unit Test Example

```python
import pytest
from unittest.mock import Mock, AsyncMock, patch

class TestCVGeneratorService:
    """Test CV generator service functionality."""
    
    def test_service_initialization(self, cv_generator_config):
        """Test that service initializes correctly."""
        service = CVGeneratorService(cv_generator_config)
        
        assert service.config == cv_generator_config
        assert service.face_provider is not None
        assert service.pdf_generator is not None
    
    @pytest.mark.asyncio
    async def test_generate_single_cv_success(self, cv_generator_service, sample_cv_template):
        """Test successful CV generation."""
        with patch.object(cv_generator_service.content_generator, 'generate_cv_content', 
                         new_callable=AsyncMock) as mock_content:
            # Test implementation
            pass
```

### Integration Test Example

```python
import pytest
from unittest.mock import patch, AsyncMock

class TestCVGenerationIntegration:
    """Test complete CV generation flow."""
    
    @pytest.mark.asyncio
    async def test_cv_generation_with_mocks(self, cv_generator_config, sample_cv_template):
        """Test CV generation with mocked external dependencies."""
        with patch('app.services.cv_generator.ThisPersonDoesNotExistProvider.get_face_image', 
                  new_callable=AsyncMock) as mock_face:
            # Test implementation
            pass
```

### API Test Example

```python
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)

class TestCVGenerationEndpoints:
    """Test CV generation endpoints."""
    
    @patch('app.api.cv_service.generate_cvs')
    def test_generate_cvs_success(self, mock_generate_cvs, client):
        """Test successful CV generation."""
        # Test implementation
        pass
```

## 🎯 Test Best Practices

### 1. **Test Organization**
- Use descriptive test class and method names
- Group related tests in classes
- Use fixtures for common setup
- Keep tests independent

### 2. **Mocking Strategy**
- Mock external dependencies (APIs, databases)
- Use `AsyncMock` for async functions
- Mock at the right level (service boundaries)
- Verify mock calls when important

### 3. **Assertions**
- Use specific assertions
- Test both success and failure cases
- Verify return values and side effects
- Test edge cases and error conditions

### 4. **Test Data**
- Use fixtures for common test data
- Create realistic test scenarios
- Avoid hardcoded values in tests
- Use factories for complex objects

### 5. **Async Testing**
- Use `@pytest.mark.asyncio` for async tests
- Mock async functions with `AsyncMock`
- Handle async context managers properly
- Test async error handling

## 🔍 Debugging Tests

### Running Tests in Debug Mode

```bash
# Run with detailed output
pytest -v -s tests/

# Run specific failing test
pytest -v -s tests/unit/test_cv_generator.py::TestCVGeneratorService::test_service_initialization

# Run with maximum verbosity
pytest -vvv tests/
```

### Common Issues

1. **Import Errors**: Ensure test files are in the correct directory structure
2. **Fixture Errors**: Check fixture names and dependencies
3. **Async Issues**: Use `@pytest.mark.asyncio` for async tests
4. **Mock Issues**: Ensure mocks are applied at the correct level

## 📈 Continuous Integration

### GitHub Actions Example

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.11
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: python run_tests.py all
      - name: Generate coverage
        run: python run_tests.py coverage
```

## 🎉 Benefits of This Test Structure

1. **Maintainability**: Well-organized, easy to find and update tests
2. **Reliability**: Comprehensive coverage of functionality
3. **Speed**: Fast unit tests, slower integration tests
4. **Clarity**: Clear separation of concerns
5. **Scalability**: Easy to add new tests and categories
6. **CI/CD Ready**: Structured for automated testing

---

**This test suite ensures the CV Engine is reliable, maintainable, and production-ready.**
